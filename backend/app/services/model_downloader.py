"""
External model storage download service.

Downloads ML model artifacts from Backblaze B2 (S3-compatible) on Render
startup when models are missing or need verification.

Usage:
    from app.services.model_downloader import ensure_models_downloaded
    ensure_models_downloaded()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Base directory for models (relative to backend/)
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Backblaze B2 environment variables
B2_ENDPOINT_URL = os.environ.get("B2_ENDPOINT_URL", "").strip()
B2_ACCESS_KEY_ID = os.environ.get("B2_ACCESS_KEY_ID", "").strip()
B2_SECRET_ACCESS_KEY = os.environ.get("B2_SECRET_ACCESS_KEY", "").strip()
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_NAME", "").strip()
B2_REGION = os.environ.get("B2_REGION", "").strip()


class DownloadError(Exception):
    """Raised when model download fails."""
    pass


class ChecksumError(Exception):
    """Raised when checksum verification fails."""
    pass


def _load_manifest() -> dict:
    """Load the model manifest."""
    manifest_path = MODELS_DIR / "model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model manifest not found at {manifest_path}")
    
    with open(manifest_path) as f:
        return json.load(f)


def _sha256_of_file(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(8192), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _get_s3_client():
    """Get S3 client configured for Backblaze B2."""
    if not all([B2_ENDPOINT_URL, B2_ACCESS_KEY_ID, B2_SECRET_ACCESS_KEY]):
        return None
    
    return boto3.client(
        's3',
        endpoint_url=B2_ENDPOINT_URL,
        aws_access_key_id=B2_ACCESS_KEY_ID,
        aws_secret_access_key=B2_SECRET_ACCESS_KEY,
        region_name=B2_REGION if B2_REGION else None,
    )


def _download_file_from_s3(object_key: str, dest_path: Path, expected_size: Optional[int] = None) -> None:
    """Download a file from Backblaze B2 to destination path."""
    if not B2_BUCKET_NAME:
        raise DownloadError(f"B2_BUCKET_NAME not configured")
    
    s3_client = _get_s3_client()
    if not s3_client:
        raise DownloadError(f"S3 client not configured - check B2 credentials")
    
    # Ensure destination directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading s3://{B2_BUCKET_NAME}/{object_key} -> {dest_path}")
    
    try:
        # Write to temporary file first
        temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
        
        # Download with progress tracking
        s3_client.download_file(
            Bucket=B2_BUCKET_NAME,
            Key=object_key,
            Filename=str(temp_path),
        )
        
        # Verify file size
        actual_size = temp_path.stat().st_size
        if expected_size and actual_size != expected_size:
            temp_path.unlink()
            raise DownloadError(
                f"Size mismatch for {dest_path.name}: "
                f"expected {expected_size}, got {actual_size}"
            )
        
        # Move to final location
        temp_path.rename(dest_path)
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        raise DownloadError(f"Failed to download {object_key}: {error_code} - {error_msg}")
    except Exception as e:
        raise DownloadError(f"Failed to download {object_key}: {e}")


def _download_with_retry(object_key: str, dest_path: Path, max_retries: int = 3) -> None:
    """Download a file with retry logic."""
    last_error = None
    for attempt in range(max_retries):
        try:
            _download_file_from_s3(object_key, dest_path)
            return
        except Exception as e:
            last_error = e
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            # Clean up any partial download
            temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
            if temp_path.exists():
                temp_path.unlink()
            if dest_path.exists():
                dest_path.unlink()
    
    raise DownloadError(f"Failed after {max_retries} attempts: {last_error}")


def _verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    """Verify file SHA256 checksum."""
    actual = _sha256_of_file(file_path)
    if actual != expected_checksum:
        logger.error(
            f"Checksum mismatch for {file_path.name}: "
            f"expected {expected_checksum}, got {actual}"
        )
        return False
    return True


def ensure_models_downloaded(force_redownload: bool = False) -> dict[str, bool]:
    """
    Ensure all required model artifacts are available locally.
    
    Downloads missing models from Backblaze B2 storage and verifies checksums.
    If models already exist with valid checksums, skips download.
    
    Args:
        force_redownload: If True, always re-download even if files exist.
    
    Returns:
        Dict mapping file paths to download status (True if downloaded, False if already existed).
    """
    # Check if B2 storage is configured
    b2_configured = all([B2_ENDPOINT_URL, B2_ACCESS_KEY_ID, B2_SECRET_ACCESS_KEY, B2_BUCKET_NAME])
    
    # If no B2 configuration, check if local models exist
    if not b2_configured:
        if MODELS_DIR.exists() and any(MODELS_DIR.rglob("*.joblib")):
            logger.info("No B2 storage configured, using local models")
            return {}
        else:
            logger.warning(
                "No B2 storage configured and no local models found. "
                "Model artifacts will not be available."
            )
            return {}
    
    # Load manifest
    try:
        manifest = _load_manifest()
    except FileNotFoundError as e:
        logger.error(f"Cannot download models: {e}")
        return {}
    
    results = {}
    total = len(manifest["artifacts"])
    logger.info(f"Ensuring {total} model artifacts are available from Backblaze B2...")
    
    for i, artifact in enumerate(manifest["artifacts"], 1):
        relative_path = artifact["relative_path"]
        dest_path = MODELS_DIR / relative_path
        expected_sha256 = artifact["sha256"]
        expected_size = artifact["size_bytes"]
        
        logger.info(f"[{i}/{total}] Checking {relative_path}...")
        
        needs_download = force_redownload
        
        if not needs_download:
            if not dest_path.exists():
                logger.info(f"  Missing, will download")
                needs_download = True
            elif not _verify_checksum(dest_path, expected_sha256):
                logger.warning(f"  Checksum invalid, will re-download")
                needs_download = True
            else:
                logger.info(f"  OK (checksum valid)")
        
        if needs_download:
            object_key = relative_path  # Use relative path as S3 object key
            try:
                _download_with_retry(object_key, dest_path, max_retries=3)
                
                # Verify after download
                if not _verify_checksum(dest_path, expected_sha256):
                    raise ChecksumError(
                        f"Checksum verification failed for {relative_path} after download"
                    )
                
                logger.info(f"  Downloaded and verified: {relative_path}")
                results[relative_path] = True
                
            except Exception as e:
                logger.error(f"  FAILED to download {relative_path}: {e}")
                results[relative_path] = False
    
    downloaded_count = sum(1 for v in results.values() if v)
    logger.info(
        f"Model download complete: {downloaded_count}/{total} files downloaded/verified"
    )
    
    return results


def get_download_status() -> dict:
    """
    Get the current download status of model artifacts.
    
    Returns:
        Dict with model availability information.
    """
    b2_configured = all([B2_ENDPOINT_URL, B2_ACCESS_KEY_ID, B2_SECRET_ACCESS_KEY, B2_BUCKET_NAME])
    
    # Check local files
    local_files = []
    missing_files = []
    
    try:
        manifest = _load_manifest()
        for artifact in manifest["artifacts"]:
            path = MODELS_DIR / artifact["relative_path"]
            if path.exists():
                if _verify_checksum(path, artifact["sha256"]):
                    local_files.append(artifact["relative_path"])
                else:
                    missing_files.append(artifact["relative_path"])
            else:
                missing_files.append(artifact["relative_path"])
    except FileNotFoundError:
        pass
    
    return {
        "storage_configured": b2_configured,
        "storage_type": "Backblaze B2" if b2_configured else "None",
        "bucket_name": B2_BUCKET_NAME if b2_configured else None,
        "local_files_count": len(local_files),
        "missing_files_count": len(missing_files),
        "missing_files": missing_files[:10] if missing_files else [],  # Limit to first 10
        "ready": len(missing_files) == 0
    }


# Module-level function to call on app startup
def initialize_models() -> None:
    """
    Initialize models on application startup.
    
    This should be called during FastAPI startup event to ensure
    models are available before serving requests.
    """
    logger.info("Initializing model artifacts...")
    results = ensure_models_downloaded()
    status = get_download_status()
    
    if status["ready"]:
        logger.info("All model artifacts ready")
    else:
        logger.warning(
            f"Model initialization incomplete: {status['missing_files_count']} files missing"
        )
    
    logger.info(f"Download status: {status}")