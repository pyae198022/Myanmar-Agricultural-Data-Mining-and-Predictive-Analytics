"""
External model storage download service.

Downloads ML model artifacts from external storage (S3-compatible) on Render
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

import httpx

logger = logging.getLogger(__name__)

# Base directory for models (relative to backend/)
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Environment variable for storage base URL
MODEL_STORAGE_BASE_URL_ENV = "MODEL_STORAGE_BASE_URL"

# Optional: credentials for private storage (S3-compatible)
MODEL_STORAGE_ACCESS_KEY = os.environ.get("MODEL_STORAGE_ACCESS_KEY", "")
MODEL_STORAGE_SECRET_KEY = os.environ.get("MODEL_STORAGE_SECRET_KEY", "")


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


def _download_file(url: str, dest_path: Path, expected_size: Optional[int] = None) -> None:
    """Download a file from URL to destination path."""
    headers = {}
    
    # Add basic auth if credentials provided
    if MODEL_STORAGE_ACCESS_KEY and MODEL_STORAGE_SECRET_KEY:
        import base64
        auth = base64.b64encode(f"{MODEL_STORAGE_ACCESS_KEY}:MODEL_STORAGE_SECRET_KEY".encode()).decode()
        headers["Authorization"] = f"Basic {auth}"
    
    # Ensure destination directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {url} -> {dest_path}")
    
    try:
        with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            
            # Write to temporary file first
            temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
            with open(temp_path, "wb") as f:
                f.write(response.content)
            
            # Verify size if expected
            if expected_size and len(response.content) != expected_size:
                temp_path.unlink()
                raise DownloadError(
                    f"Size mismatch for {dest_path.name}: "
                    f"expected {expected_size}, got {len(response.content)}"
                )
            
            # Move to final location
            temp_path.rename(dest_path)
            
    except httpx.HTTPError as e:
        raise DownloadError(f"Failed to download {url}: {e}")


def _download_with_retry(url: str, dest_path: Path, max_retries: int = 3) -> None:
    """Download a file with retry logic."""
    last_error = None
    for attempt in range(max_retries):
        try:
            _download_file(url, dest_path)
            return
        except Exception as e:
            last_error = e
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
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
    
    Downloads missing models from external storage and verifies checksums.
    If models already exist with valid checksums, skips download.
    
    Args:
        force_redownload: If True, always re-download even if files exist.
    
    Returns:
        Dict mapping file paths to download status (True if downloaded, False if already existed).
    """
    # Check if storage URL is configured
    storage_base_url = os.environ.get(MODEL_STORAGE_BASE_URL_ENV, "").strip()
    
    # If no storage URL, check if local models exist
    if not storage_base_url:
        if MODELS_DIR.exists() and any(MODELS_DIR.rglob("*.joblib")):
            logger.info("No MODEL_STORAGE_BASE_URL configured, using local models")
            return {}
        else:
            logger.warning(
                "No MODEL_STORAGE_BASE_URL configured and no local models found. "
                "Model artifacts will not be available."
            )
            return {}
    
    # Remove trailing slash
    storage_base_url = storage_base_url.rstrip("/")
    
    # Load manifest
    try:
        manifest = _load_manifest()
    except FileNotFoundError as e:
        logger.error(f"Cannot download models: {e}")
        return {}
    
    results = {}
    total = len(manifest["artifacts"])
    logger.info(f"Ensuring {total} model artifacts are available...")
    
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
            url = f"{storage_base_url}/{relative_path}"
            try:
                _download_with_retry(url, dest_path, max_retries=3)
                
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
    storage_url = os.environ.get(MODEL_STORAGE_BASE_URL_ENV, "").strip()
    has_storage = bool(storage_url)
    
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
        "storage_configured": has_storage,
        "storage_url": storage_url[:50] + "..." if len(storage_url) > 50 else storage_url if storage_url else None,
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