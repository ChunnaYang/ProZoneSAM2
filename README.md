# Medical SAM2 Interactive Demo

An interactive web application for medical image segmentation using Medical SAM2 (Segment Anything Model 2 for Medical Imaging).

## Features

### 2D Segmentation (Homepage)
- **Interactive Image Upload**: Upload medical images in common formats (PNG, JPG, etc.)
- **Box-based Segmentation**: Click and drag to draw bounding boxes for WG and CG regions
- **Real-time Visualization**: View segmentation masks (WG, CG, PZ) overlaid on the original image
- **Access**: http://localhost:5000/

### 3D Segmentation (Prostate MRI) ⭐ New!
- **3D Volume Support**: Upload .nii.gz format MRI volumes
- **Multi-slice Navigation**: Browse through all slices with manual/auto controls
- **3D Segmentation**: Perform segmentation on selected slices with bounding box prompts
- **Multi-region Output**: Visualize CG (Central Gland) and PZ (Peripheral Zone) regions
- **Access**: http://localhost:5000/prostate-3d

**⚠️ Important File Format Requirements:**
- **Only supports `.nii.gz` files** (gzip-compressed NIfTI format)
- **Does NOT support**: `.nii` (uncompressed), `.dcm` (DICOM), or image files
- See [USER_GUIDE.md](USER_GUIDE.md) for detailed instructions on file preparation

**File Upload Validation:**
- System automatically validates file format
- Checks gzip magic number to ensure valid .nii.gz files
- Provides specific error messages for invalid formats
- Warns for large files (>500MB) but still allows upload

### General Features
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Built with Next.js 16, React 19, and shadcn/ui components
- **GPU/CPU Support**: Automatically detects and uses GPU if available

## Getting Started

### Prerequisites

- Node.js 20+
- Python 3.8+ (for model inference)
- pnpm (package manager)

### Installation

#### 1. Install Node.js dependencies
```bash
pnpm install
```

#### 2. Install Python dependencies
```bash
# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install medical imaging dependencies
pip install monai hydra-core omegaconf nibabel scipy
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

#### 3. Start the development server
```bash
pnpm dev
```

#### 4. Open [http://localhost:5000](http://localhost:5000) in your browser

### Python Dependencies

The system requires the following Python packages:
- **PyTorch** 2.10.0+ (for model inference)
- **Nibabel** (for loading .nii.gz files)
- **SciPy** (for morphological post-processing)
- **MONAI** (medical imaging utilities)
- **Hydra** (configuration management)

See [DEPENDENCIES_INSTALLATION.md](DEPENDENCIES_INSTALLATION.md) for detailed installation instructions.

### Current System Status

✅ **Installed Dependencies:**
- PyTorch 2.9.1+cpu (CPU version)
- torchvision 0.24.1+cpu
- Pillow 10.4.0
- numpy 2.4.1

⚠️ **Not Installed:**
- segment-anything-2 package
- Model checkpoint files

**Current Mode:** Mock mode (generates placeholder masks without actual segmentation)

**To enable real model inference:** See [MODEL_SETUP.md](MODEL_SETUP.md) for detailed instructions.

## Usage

1. **Upload an Image**: Click the "Upload Image" button and select a medical image
2. **Draw a Box**: Click and drag on the image to draw a bounding box around the region you want to segment
3. **Run Segmentation**: Click "Run Segmentation" to process the image
4. **View Results**: The segmentation mask will be overlaid on the original image

## Quick Start (Mock Mode)

The demo currently runs in mock mode and works immediately without any additional setup:

1. Visit http://localhost:5000
2. Upload an image
3. Draw a bounding box by clicking and dragging
4. Click "Run Segmentation"
5. View the generated placeholder mask

The mock mode demonstrates the user interface and interaction flow without requiring a trained model.

## Integration with Medical SAM2 Model

To use the actual SAM2 model for real segmentation, follow these steps:

### Step 1: Install SAM2 Package

```bash
pip install segment-anything-2
```

Or clone from source:
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd ..
```

### Step 2: Download Model Checkpoint

Choose a model based on your needs:

```bash
# Base model (recommended for testing)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

# Large model (higher accuracy)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# Small model (faster)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt
```

### Step 3: Configure Model Path

Edit `.env.local`:
```bash
MEDICAL_SAM2_CHECKPOINT=sam2_hiera_base_plus.pt
MEDICAL_SAM2_CONFIG=sam2_hiera_b+.yaml
USE_PYTHON_SEGMENTATION=true
```

### Step 4: Restart the Server

```bash
# Stop the current server (Ctrl+C)
pnpm dev
```

## Project Structure

```
.
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   └── segment/
│   │   │       └── route.ts      # API endpoint for segmentation
│   │   ├── components/
│   │   │   └── ui/               # shadcn/ui components
│   │   ├── page.tsx              # Main page component
│   │   └── layout.tsx            # Root layout
│   └── ...
├── scripts/
│   └── segment.py                # Python script for SAM2 inference
├── .coze                         # Configuration file
├── package.json
└── README.md
```

## API Endpoint

### POST /api/segment

Request body:
```json
{
  "image": "data:image/jpeg;base64,...",
  "box": {
    "x": 100,
    "y": 100,
    "width": 200,
    "height": 200
  }
}
```

Response:
```json
{
  "success": true,
  "mask": "data:image/png;base64,..."
}
```

## Customization

### Changing the Mask Color

Edit the `create_mock_mask()` function in `scripts/segment.py` or update the mask generation in the actual model inference.

### Adding Multiple Mask Outputs

Modify the API to return multiple masks when using `multimask_output=True` in SAM2.

### Supporting Different Input Formats

Extend the image loading logic to support DICOM, NIfTI, or other medical imaging formats.

## Troubleshooting

### Demo shows "Mock Mode - Model Not Loaded"

**Cause:** The SAM2 package is not installed or the model checkpoint is missing.

**Solution:**
1. Install segment-anything-2:
   ```bash
   pip install segment-anything-2
   ```
2. Download a model checkpoint (see Quick Start above)
3. Set `USE_PYTHON_SEGMENTATION=true` in `.env.local`

### 3D Volume Upload Errors (`.nii.gz` files)

**Error:** "File format error: Please upload a valid .nii.gz file"

**Cause:** The uploaded file is not a valid gzip-compressed NIfTI file.

**Solution:**
1. **Check file extension**: Must be `.nii.gz` (not `.nii` or other formats)
2. **Verify file integrity**: Ensure the file is properly compressed with gzip
3. **Re-export the file**: Use medical imaging software (ITK-SNAP, 3D Slicer) to export as `.nii.gz`
4. **Convert from DICOM**: If you have DICOM files, use `dcm2niix` to convert:
   ```bash
   dcm2niix -o output_dir input_dicom_folder
   ```
5. **Validate with Python**:
   ```python
   import nibabel as nib
   nii = nib.load('your_file.nii.gz')
   print(nii.shape)  # Should print the volume dimensions
   ```

**Error:** "File too small, not a valid NIfTI file"

**Cause:** The file is corrupted or too small to be a valid MRI volume.

**Solution:**
- Re-download or re-export the file
- Check if the file was truncated during transfer
- Verify the original file size (typical MRI volumes are 10-500MB)

**Error:** "Uploaded file is not a valid .nii.gz (gzip compressed NIfTI) file"

**Cause:** The file is not compressed with gzip or is not a NIfTI format.

**Solution:**
- Convert NIfTI files to gzip-compressed format:
  ```bash
  gzip -c input.nii > output.nii.gz
  ```
- Use nibabel to validate and convert:
  ```python
  import nibabel as nib
  nii = nib.load('input.nii')
  nib.save(nii, 'output.nii.gz')
  ```

For more detailed troubleshooting, see [USER_GUIDE.md](USER_GUIDE.md).

### Python Script Not Running

- Ensure Python 3 is installed: `python3 --version`
- Check the environment variable: `echo $USE_PYTHON_SEGMENTATION`
- Verify the script is executable: `ls -l scripts/segment.py`

### Model Not Loading

**Error:** "Model checkpoint not found"

**Solution:**
- Verify the checkpoint file path in `.env.local`
- Download the model file to the specified location
- Check file permissions: `ls -l sam2_hiera_base_plus.pt`

**Error:** CUDA out of memory

**Solution:**
- Use the CPU version (already installed)
- Use a smaller model: `sam2_hiera_small.pt`
- Reduce input image resolution

### Frontend Issues

- **Page not loading:** Check that port 5000 is available
- **White screen:** Open browser DevTools (F12) and check Console for errors
- **API errors:** Check the terminal output for server-side errors

### Segmentation Results Not Accurate

**Suggestions:**
- Try different box positions and sizes
- Use a larger model (base_plus or large)
- For medical images, consider using Medical SAM2 instead of standard SAM2
- Ensure good lighting and contrast in the input image

### Performance Issues

**Slow segmentation:**
- Use the small model: `sam2_hiera_small.pt`
- Reduce image size before uploading
- Use GPU acceleration (requires CUDA-compatible GPU)

### Network Issues

**Can't download model files:**
- Try alternative download mirrors
- Use a VPN or proxy if accessing from restricted regions
- Request the model files from a colleague

## Advanced Configuration

### Using GPU Acceleration

If you have an NVIDIA GPU with CUDA:

```bash
# Uninstall CPU version
pip uninstall torch torchvision

# Install CUDA version
pip install torch torchvision
```

### Custom Model Configuration

To use a custom model path, edit `.env.local`:
```bash
MEDICAL_SAM2_CHECKPOINT=/path/to/your/model.pt
MEDICAL_SAM2_CONFIG=/path/to/your/config.yaml
USE_PYTHON_SEGMENTATION=true
```

### Batch Processing

The Python script can be modified to process multiple boxes at once. See `scripts/segment.py` for details.

## Support and Resources

- **SAM2 Official Repository:** [github.com/facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
- **Medical SAM2:** [github.com/bowang-lab/Medical-SAM2](https://github.com/bowang-lab/Medical-SAM2)
- **Detailed Setup Guide:** See [MODEL_SETUP.md](MODEL_SETUP.md)
- **Next.js Documentation:** [nextjs.org/docs](https://nextjs.org/docs)
- **shadcn/ui Components:** [ui.shadcn.com](https://ui.shadcn.com/)

## License

This project is for demonstration purposes. Medical SAM2 and SAM2 have their own licenses.
