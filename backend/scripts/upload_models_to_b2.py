#!/usr/bin/env python3
"""
Upload ML model artifacts to Backblaze B2 (S3-compatible) storage.

This script reads model_manifest.json, verifies local files and checksums,
then uploads all required artifacts to a Backblaze B2 bucket.

Usage:
    python upload_models_to_b2.py [--dry-run] [--force]

Environment variables required:
    B2_ENDPOINT_URL: Backblaze B2 S3 endpoint URL
    B2_ACCESS_KEY_ID: B2 Application Key ID (20 characters)
    B2_SECRET_ACCESS_KEY: B2 Application Key (40 characters)
    B2_BUCKET_NAME: B2 bucket name
    B2_REGION: B2 region (optional, usually included in endpoint)

Examples:
    # Dry run (verify files but don't upload)
    python upload_models_to_b2.py --dry-run

    # Upload all models
    python upload_models_to_b2.py

    # Force upload even if checksum verification fails (not recommended)
    python upload_models_to_b2.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
BACKEND_DIR = Path(__file__).parent.parent
MODELS_DIR = BACKEND_DIR / "models"
MANIFEST_PATH = MODELS_DIR / "model_manifest.json"

# Environment variables
B2_ENDPOINT_URL = os.environ.get("B2_ENDPOINT_URL", "").strip()
B2_ACCESS_KEY_ID = os.environ.get("B2_ACCESS_KEY_ID", "").strip()
B2_SECRET_ACCESS_KEY = os.environ.get("B2_SECRET_ACCESS_KEY", "").strip()
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_NAME", "").strip()
B2_REGION = os.environ.get("B2_REGION", "").strip()


class UploadError(Exception):
    """Raised when model upload fails."""
    pass


class VerificationError(Exception):
    """Raised when file verification fails."""
    pass


def calculate_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_local_file(file_path: Path, expected_sha256: str, expected_size: int) -> tuple[bool, str]:
    """
    Verify a local file against expected checksum and size.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"
    
    actual_size = file_path.stat().st_size
    if actual_size != expected_size:
        return False, f"Size mismatch: expected {expected_size}, got {actual_size}"
    
    try:
        actual_sha256 = calculate_sha256(file_path)
        if actual_sha256 != expected_sha256:
            return False, f"Checksum mismatch: expected {expected_sha256[:16]}..., got {actual_sha256[:16]}..."
    except Exception as e:
        return False, f"Failed to calculate checksum: {e}"
    
    return True, "OK"


def get_s3_client():
    """Get S3 client configured for Backblaze B2."""
    if not all([B2_ENDPOINT_URL, B2_ACCESS_KEY_ID, B2_SECRET_ACCESS_KEY]):
        raise UploadError("Missing B2 credentials. Please set B2_ENDPOINT_URL, B2_ACCESS_KEY_ID, and B2_SECRET_ACCESS_KEY.")
    
    if not B2_BUCKET_NAME:
        raise UploadError("Missing B2_BUCKET_NAME.")
    
    return boto3.client(
        's3',
        endpoint_url=B2_ENDPOINT_URL,
        aws_access_key_id=B2_ACCESS_KEY_ID,
        aws_secret_access_key=B2_SECRET_ACCESS_KEY,
        region_name=B2_REGION if B2_REGION else None,
    )


def check_bucket_exists(s3_client, bucket_name: str) -> bool:
    """Check if the B2 bucket exists and is accessible."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == '404':
            logger.error(f"Bucket '{bucket_name}' does not exist.")
        elif error_code == '403':
            logger.error(f"Access denied to bucket '{bucket_name}'. Check credentials and permissions.")
        else:
            logger.error(f"Error accessing bucket '{bucket_name}': {e}")
        return False


def upload_file_to_b2(s3_client, bucket_name: str, file_path: Path, object_key: str, dry_run: bool = False) -> bool:
    """Upload a file to Backblaze B2 with progress tracking."""
    file_size = file_path.stat().st_size
    
    if dry_run:
        logger.info(f"[DRY RUN] Would upload: {file_path} -> s3://{bucket_name}/{object_key} ({file_size:,} bytes)")
        return True
    
    try:
        logger.info(f"Uploading {file_path} -> s3://{bucket_name}/{object_key} ({file_size:,} bytes)")
        
        # Use upload_file with progress callback
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=file_path.name) as pbar:
            def progress_callback(bytes_transferred):
                pbar.update(bytes_transferred - pbar.n)
            
            s3_client.upload_file(
                Filename=str(file_path),
                Bucket=bucket_name,
                Key=object_key,
                Callback=progress_callback
            )
        
        # Verify uploaded object
        response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        uploaded_size = response['ContentLength']
        
        if uploaded_size != file_size:
            logger.error(f"Size verification failed for {object_key}: uploaded {uploaded_size}, expected {file_size}")
            return False
        
        logger.info(f"✓ Uploaded and verified: {object_key}")
        return True
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        logger.error(f"Failed to upload {object_key}: {error_code} - {error_msg}")
        return False
    except Exception as e:
        logger.error(f"Failed to upload {object_key}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload ML model artifacts to Backblaze B2")
    parser.add_argument("--dry-run", action="store_true", help="Verify files but don't upload")
    parser.add_argument("--force", action="store_true", help="Upload even if checksum verification fails (not recommended)")
    parser.add_argument("--skip-verification", action="store_true", help="Skip checksum verification before upload")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Backblaze B2 Model Upload Script")
    logger.info("=" * 60)
    
    # Check environment variables
    required_env_vars = ["B2_ENDPOINT_URL", "B2_ACCESS_KEY_ID", "B2_SECRET_ACCESS_KEY", "B2_BUCKET_NAME"]
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables or add them to your .env file.")
        sys.exit(1)
    
    # Show configuration (mask secrets)
    logger.info(f"B2 Endpoint: {B2_ENDPOINT_URL}")
    logger.info(f"B2 Bucket: {B2_BUCKET_NAME}")
    logger.info(f"B2 Region: {B2_REGION or '(not specified)'}")
    logger.info(f"Access Key ID: {B2_ACCESS_KEY_ID[:4]}...{B2_ACCESS_KEY_ID[-4:] if len(B2_ACCESS_KEY_ID) > 8 else ''}")
    logger.info(f"Models Directory: {MODELS_DIR}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info(f"Force Upload: {args.force}")
    logger.info("-" * 60)
    
    # Load manifest
    if not MANIFEST_PATH.exists():
        logger.error(f"Manifest file not found: {MANIFEST_PATH}")
        sys.exit(1)
    
    try:
        with open(MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse manifest JSON: {e}")
        sys.exit(1)
    
    artifacts = manifest.get("artifacts", [])
    total_count = len(artifacts)
    total_size = manifest.get("total_size_bytes", 0)
    
    logger.info(f"Found {total_count} artifacts in manifest ({total_size:,} bytes total)")
    logger.info("-" * 60)
    
    # Verify local files
    logger.info("Verifying local model files...")
    verification_results = []
    failed_verifications = []
    
    for i, artifact in enumerate(artifacts, 1):
        relative_path = artifact["relative_path"]
        file_path = MODELS_DIR / relative_path
        expected_sha256 = artifact["sha256"]
        expected_size = artifact["size_bytes"]
        
        if args.skip_verification:
            is_valid, message = True, "Skipped"
        else:
            is_valid, message = verify_local_file(file_path, expected_sha256, expected_size)
        
        status = "✓" if is_valid else "✗"
        logger.info(f"[{i}/{total_count}] {status} {relative_path}: {message}")
        
        verification_results.append({
            "artifact": artifact,
            "file_path": file_path,
            "is_valid": is_valid,
            "message": message,
        })
        
        if not is_valid and not args.force:
            failed_verifications.append(relative_path)
    
    if failed_verifications and not args.force:
        logger.error(f"\n{len(failed_verifications)} files failed verification:")
        for f in failed_verifications[:5]:  # Show first 5
            logger.error(f"  - {f}")
        if len(failed_verifications) > 5:
            logger.error(f"  ... and {len(failed_verifications) - 5} more")
        logger.error("\nUse --force to upload anyway (not recommended)")
        sys.exit(1)
    
    valid_count = sum(1 for r in verification_results if r["is_valid"])
    logger.info(f"\nVerification complete: {valid_count}/{total_count} files valid")
    
    if args.dry_run:
        logger.info("\nDry run completed successfully. No files were uploaded.")
        logger.info("To upload files, run without --dry-run flag.")
        sys.exit(0)
    
    # Connect to B2
    logger.info("\nConnecting to Backblaze B2...")
    try:
        s3_client = get_s3_client()
    except UploadError as e:
        logger.error(f"Failed to create S3 client: {e}")
        sys.exit(1)
    
    # Check bucket exists
    if not check_bucket_exists(s3_client, B2_BUCKET_NAME):
        sys.exit(1)
    
    # Upload files
    logger.info(f"\nUploading {len(verification_results)} files to Backblaze B2...")
    upload_results = []
    uploaded_count = 0
    failed_count = 0
    
    for i, result in enumerate(verification_results, 1):
        artifact = result["artifact"]
        file_path = result["file_path"]
        relative_path = artifact["relative_path"]
        
        logger.info(f"\n[{i}/{total_count}] Processing: {relative_path}")
        
        if not result["is_valid"] and not args.force:
            logger.warning(f"  Skipping (verification failed: {result['message']})")
            upload_results.append({
                "artifact": artifact,
                "success": False,
                "error": f"Verification failed: {result['message']}"
            })
            failed_count += 1
            continue
        
        # Upload file
        success = upload_file_to_b2(
            s3_client=s3_client,
            bucket_name=B2_BUCKET_NAME,
            file_path=file_path,
            object_key=relative_path,
            dry_run=args.dry_run
        )
        
        if success:
            upload_results.append({"artifact": artifact, "success": True})
            uploaded_count += 1
        else:
            upload_results.append({
                "artifact": artifact,
                "success": False,
                "error": "Upload failed"
            })
            failed_count += 1
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("UPLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total artifacts: {total_count}")
    logger.info(f"Successfully uploaded: {uploaded_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Total size: {total_size:,} bytes ({total_size / (1024**3):.2f} GB)")
    
    if failed_count > 0:
        logger.error(f"\nFailed uploads:")
        for result in upload_results:
            if not result["success"]:
                artifact = result["artifact"]
                error = result.get("error", "Unknown error")
                logger.error(f"  - {artifact['relative_path']}: {error}")
        sys.exit(1)
    
    logger.info("\n✓ All model artifacts successfully uploaded to Backblaze B2!")
    logger.info(f"Bucket: {B2_BUCKET_NAME}")
    logger.info(f"Endpoint: {B2_ENDPOINT_URL}")
    
    # Generate sample environment config
    logger.info("\n" + "=" * 60)
    logger.info("DEPLOYMENT CONFIGURATION")
    logger.info("=" * 60)
    logger.info("Add these environment variables to your Render deployment:")
    logger.info("")
    logger.info(f"B2_ENDPOINT_URL={B2_ENDPOINT_URL}")
    logger.info(f"B2_ACCESS_KEY_ID={B2_ACCESS_KEY_ID}")
    logger.info(f"B2_SECRET_ACCESS_KEY=[your_secret_key]")
    logger.info(f"B2_BUCKET_NAME={B2_BUCKET_NAME}")
    if B2_REGION:
        logger.info(f"B2_REGION={B2_REGION}")
    logger.info("")
    logger.info("Note: Never commit secrets to version control!")


if __name__ == "__main__":
    main()