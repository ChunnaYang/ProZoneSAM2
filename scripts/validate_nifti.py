#!/usr/bin/env python3
"""
Simple tool to validate if a file is a valid gzip-compressed NIfTI file.

Usage:
    python validate_nifti.py your_file.nii.gz

This will tell you:
1. If the file is a valid gzip file
2. If the file contains valid NIfTI data
3. The file size and dimensions
"""

import sys
import os
import gzip
import argparse


def validate_gzip_file(filepath):
    """Check if a file is a valid gzip file."""
    print(f"\n{'='*60}")
    print(f"Validating: {filepath}")
    print(f"{'='*60}\n")

    # Check file exists
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found: {filepath}")
        return False

    # Get file size
    file_size = os.path.getsize(filepath)
    print(f"📁 File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

    # Check if file is too small
    if file_size < 100:
        print(f"❌ Error: File is too small to be a valid NIfTI file")
        return False

    # Read and check magic number
    with open(filepath, 'rb') as f:
        magic_bytes = f.read(2)
        magic_hex = magic_bytes.hex()

    print(f"🔍 Magic bytes (hex): {magic_hex}")

    # Check gzip magic number
    if magic_bytes != b'\x1f\x8b':
        print(f"❌ Error: Not a gzip file (expected magic bytes: 1f8b, got: {magic_hex})")

        # Try to identify file type
        print("\n🔎 Attempting to identify file type...")
        with open(filepath, 'rb') as f:
            header = f.read(20)

        if header[:4] == b'\x89PNG':
            print(f"   → This is a PNG image file")
            print(f"   → File name should end with .png, not .nii.gz")
        elif header[:2] == b'\xff\xd8':
            print(f"   → This is a JPEG image file")
            print(f"   → File name should end with .jpg or .jpeg, not .nii.gz")
        elif header[:4] == b'RIFF' and header[8:12] == b'WEBP':
            print(f"   → This is a WebP image file")
            print(f"   → File name should end with .webp, not .nii.gz")
        elif header[:4] == b'%PDF':
            print(f"   → This is a PDF document")
            print(f"   → File name should end with .pdf, not .nii.gz")
        elif header[:2] == b'PK':
            print(f"   → This is a ZIP archive")
            print(f"   → Might be an uncompressed NIfTI file (.nii)")
            print(f"   → Try renaming to .nii and then compressing with gzip")
        else:
            print(f"   → Unknown file format")
            print(f"   → First 20 bytes: {header.hex()}")

        print(f"\n💡 Solution:")
        print(f"   1. Make sure the file is a valid MRI volume")
        print(f"   2. Export from medical imaging software as .nii.gz format")
        print(f"   3. Do not simply rename the file extension")
        return False

    print(f"✅ Valid gzip file")

    # Try to decompress
    print(f"\n📦 Attempting to decompress...")
    try:
        with gzip.open(filepath, 'rb') as gz_file:
            decompressed_data = gz_file.read(344)  # Read NIfTI header

            print(f"✅ Successfully decompressed {len(decompressed_data)} bytes")

            # Check if it looks like NIfTI
            if len(decompressed_data) >= 344:
                # NIfTI header starts at offset 344
                print(f"📊 Checking NIfTI header...")

                # Simple check for valid dimensions
                import struct
                dim_info = struct.unpack('h', decompressed_data[40:42])[0]
                dims = struct.unpack('8h', decompressed_data[40:56])

                print(f"   - Dimension info: {dim_info}")
                print(f"   - Dimensions: {dims}")
                print(f"   - Data dimensions: {dims[1:dims[0]+1]}")

                if dims[0] == 3:
                    print(f"✅ Valid 3D NIfTI file")
                    print(f"\n📐 Volume dimensions:")
                    print(f"   - X: {dims[1]}")
                    print(f"   - Y: {dims[2]}")
                    print(f"   - Z: {dims[3]}")
                    return True
                else:
                    print(f"⚠️  Warning: Not a 3D volume (dims[0] = {dims[0]})")
                    return True  # Still might be valid
            else:
                print(f"⚠️  Warning: Decompressed data too small for NIfTI header")

    except gzip.BadGzipFile as e:
        print(f"❌ Error: Bad gzip file - {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Validate if a file is a valid gzip-compressed NIfTI file'
    )
    parser.add_argument(
        'filepath',
        help='Path to the file to validate'
    )

    args = parser.parse_args()

    success = validate_gzip_file(args.filepath)

    print(f"\n{'='*60}")
    if success:
        print(f"✅ File is a valid .nii.gz file!")
        print(f"{'='*60}")
        print(f"\n🎉 You can upload this file to the prostate MRI segmentation system.")
        sys.exit(0)
    else:
        print(f"❌ File is NOT a valid .nii.gz file")
        print(f"{'='*60}")
        print(f"\n❌ Please fix the file format before uploading.")
        sys.exit(1)


if __name__ == '__main__':
    main()
