import { NextRequest, NextResponse } from 'next/server';

interface LoadVolumeRequestBody {
  volume: string;  // Base64 encoded nii.gz file
}

interface LoadVolumeResponse {
  success: boolean;
  shape?: [number, number, number];  // [D, H, W]
  slices?: string[];  // Array of base64 encoded slice images
  error?: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: LoadVolumeRequestBody = await request.json();
    const { volume } = body;

    // Validate input
    if (!volume) {
      return NextResponse.json(
        { success: false, error: 'Volume data is required' },
        { status: 400 }
      );
    }

    // Load and process the volume
    const loadResult = await loadNiftiVolume(volume);

    if (loadResult.error) {
      return NextResponse.json(
        { success: false, error: loadResult.error },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      shape: loadResult.shape,
      slices: loadResult.slices,
    });
  } catch (error) {
    console.error('Volume loading error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

async function loadNiftiVolume(
  volumeBase64: string
): Promise<{ shape?: [number, number, number]; slices?: string[]; error?: string }> {
  const { spawn } = require('child_process');

  return new Promise((resolve, reject) => {
    const inputData = JSON.stringify({ volume: volumeBase64 });

    // Prepare environment variables
    const env = {
      ...process.env,
      PYTHONPATH: '/usr/local/lib/python3.12/dist-packages:/workspace/projects/Seg-code-try2region-noise',
    };

    // Use a helper script to load and preview volume
    const pythonProcess = spawn('/usr/bin/python3', ['-c', loadVolumeScript], {
      env: env,
      cwd: '/workspace/projects',
    });

    // Write data to stdin
    pythonProcess.stdin.write(inputData);
    pythonProcess.stdin.end();

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data: Buffer) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data: Buffer) => {
      stderr += data.toString();
      console.log('[Load Volume] Python stderr:', data.toString().substring(0, 200));
    });

    pythonProcess.on('close', (code: number) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdout);
          if (result.success) {
            resolve({ shape: result.shape, slices: result.slices });
          } else {
            resolve({ error: result.error || 'Failed to load volume' });
          }
        } catch (e) {
          resolve({ error: `Failed to parse output: ${e instanceof Error ? e.message : String(e)}` });
        }
      } else {
        resolve({ error: `Python script exited with code ${code}: ${stderr}` });
      }
    });

    pythonProcess.on('error', (err: Error) => {
      resolve({ error: `Failed to start Python: ${err.message}` });
    });
  });
}

// Embedded Python script to load and preview NIfTI volume
const loadVolumeScript = `
import sys
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image

try:
    import nibabel as nib
except ImportError:
    print(json.dumps({"success": False, "error": "nibabel not installed"}))
    sys.exit(1)

try:
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    volume_b64 = input_data.get("volume")

    if not volume_b64:
        print(json.dumps({"success": False, "error": "No volume data provided"}))
        sys.exit(1)

    # Decode base64
    if ',' in volume_b64:
        volume_b64 = volume_b64.split(',')[1]

    nifti_data = base64.b64decode(volume_b64)
    
    # Check if file size is reasonable
    if len(nifti_data) < 100:
        print(json.dumps({"success": False, "error": f"File too small ({len(nifti_data)} bytes), not a valid NIfTI file"}))
        sys.exit(1)
    
    # Verify gzip magic number
    if nifti_data[:2] != b'\\x1f\\x8b':
        print(f"[ERROR] Invalid gzip file format. Magic bytes: {nifti_data[:2].hex()}", file=sys.stderr)
        print(json.dumps({"success": False, "error": "Uploaded file is not a valid .nii.gz (gzip compressed NIfTI) file. Please upload a file with .nii.gz extension."}))
        sys.exit(1)

    # Save to temp file
    temp_path = '/tmp/temp_volume_load.nii.gz'
    with open(temp_path, 'wb') as f:
        f.write(nifti_data)

    # Load with nibabel
    nii_img = nib.load(temp_path)
    data = nii_img.get_fdata()

    # Clean up
    import os
    os.remove(temp_path)

    print(f"[INFO] Loaded volume with original shape: {data.shape}", file=sys.stderr)

    # For prostate MRI, the data is typically (H, W, D)
    # We want to display axial slices (along the last dimension)
    # If the first dimension is the smallest, it might be the slice dimension
    shape = data.shape

    # Determine which dimension to use for slices (usually the smallest dimension)
    # For most prostate MRI: shape should be (H, W, D) where D is the number of axial slices
    if shape[0] == min(shape):
        # First dimension is the slice dimension, transpose to (D, H, W)
        print(f"[INFO] First dimension is smallest, transposing from {shape} to ({shape[0]}, {shape[1]}, {shape[2]})", file=sys.stderr)
        data = np.transpose(data, (0, 1, 2))
    elif shape[2] == min(shape) or shape[0] > shape[2]:
        # Third dimension is likely the slice dimension
        print(f"[INFO] Using third dimension as slice dimension: {shape}", file=sys.stderr)
        # Data is already in (H, W, D) format, which is what we want
        pass
    else:
        # Try to reorder dimensions
        print(f"[INFO] Attempting to reorder dimensions from {shape}", file=sys.stderr)

    # Ensure data is in (H, W, D) format for prostate MRI axial views
    # Axial slices are along the D (depth) dimension
    if data.ndim == 3:
        H, W, D = data.shape
        print(f"[INFO] Final data shape: (H={H}, W={W}, D={D}) - D is axial slice dimension", file=sys.stderr)
    else:
        print(f"[ERROR] Unexpected data dimensions: {data.shape}", file=sys.stderr)
        print(json.dumps({"success": False, "error": f"Unexpected data dimensions: {data.shape}"}))
        sys.exit(1)

    # Normalize volume
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8) * 255
    data_norm = data_norm.astype(np.uint8)

    # Convert all axial slices to base64
    slices = []
    for i in range(D):
        slice_img = data_norm[:, :, i]  # Extract axial slice
        pil_img = Image.fromarray(slice_img, mode='L')
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        slice_b64 = base64.b64encode(buffer.getvalue()).decode()
        slices.append(f"data:image/png;base64,{slice_b64}")

    result = {
        "success": True,
        "shape": [D, H, W],  # Return as (D, H, W) for the UI
        "slices": slices
    }
    print(json.dumps(result))

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}), file=sys.stderr)
    sys.exit(1)
`;
