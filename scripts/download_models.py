#!/usr/bin/env python3
"""
Download models from URLs for Railway deployment.

Set these environment variables:
- MEDICAL_MODEL_URL: URL to download best_mean3d_model.pth
- SAM_MODEL_URL: URL to download sam2_hiera_small.pt

Example:
    MEDICAL_MODEL_URL=https://example.com/models/best_mean3d_model.pth \
    SAM_MODEL_URL=https://example.com/models/sam2_hiera_small.pt \
    python3 scripts/download_models.py
"""

import os
import sys

def download_file(url: str, dest_path: str) -> bool:
    """Download a file from URL."""
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return True
    
    import urllib.request
    import urllib.error
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    print(f"Downloading: {url}")
    print(f"Destination: {dest_path}")
    
    class DownloadProgress:
        def __init__(self):
            self.bytes_downloaded = 0
        
        def __call__(self, block_num, block_size, total_size):
            self.bytes_downloaded += block_size
            mb = self.bytes_downloaded / (1024 * 1024)
            if total_size > 0:
                total_mb = total_size / (1024 * 1024)
                print(f"\r  Downloaded: {mb:.1f}MB / {total_mb:.1f}MB", end='', flush=True)
            else:
                print(f"\r  Downloaded: {mb:.1f}MB", end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=DownloadProgress().__call__)
        print()  # New line after progress
        print(f"Download complete: {dest_path}")
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def main():
    # Paths
    project_dir = '/workspace/projects/Seg-code-try2region-noise'
    work_dir = os.path.join(project_dir, 'work_dir/sam2_hiera_s_20251024_191552')
    checkpoint_dir = os.path.join(project_dir, 'checkpoints')
    
    medical_model_path = os.path.join(work_dir, 'best_mean3d_model.pth')
    sam_model_path = os.path.join(checkpoint_dir, 'sam2_hiera_small.pt')
    
    success = True
    
    # Download medical model
    medical_url = os.environ.get('MEDICAL_MODEL_URL')
    if medical_url:
        if not download_file(medical_url, medical_model_path):
            success = False
    else:
        print("MEDICAL_MODEL_URL not set, skipping medical model download")
    
    # Download SAM base model
    sam_url = os.environ.get('SAM_MODEL_URL')
    if sam_url:
        if not download_file(sam_url, sam_model_path):
            success = False
    else:
        print("SAM_MODEL_URL not set, skipping SAM model download")
    
    if success:
        print("\nAll model downloads completed successfully!")
    else:
        print("\nSome model downloads failed. Check errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
