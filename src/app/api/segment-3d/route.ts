import { NextRequest, NextResponse } from 'next/server';

interface Box3D {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  type: 'WG' | 'CG';
  slice_idx: number;  // Slice index in the 3D volume
}

interface Segment3DRequestBody {
  volume: string;  // Base64 encoded nii.gz file
  boxes: Box3D[];
  useMedical?: boolean;
}

interface Segment3DResponse {
  success: boolean;
  results?: Record<string, {  // Key is slice index
    WG?: string;
    CG?: string;
    PZ?: string;
  }>;
  error?: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: Segment3DRequestBody = await request.json();
    const { volume, boxes, useMedical } = body;

    // Validate input
    if (!volume || !boxes || boxes.length === 0) {
      return NextResponse.json(
        { success: false, error: 'Volume and at least one box are required' },
        { status: 400 }
      );
    }

    // Validate all boxes
    for (const box of boxes) {
      if (box.width <= 0 || box.height <= 0) {
        return NextResponse.json(
          { success: false, error: 'All boxes must have positive width and height' },
          { status: 400 }
        );
      }
      if (box.slice_idx === undefined || box.slice_idx < 0) {
        return NextResponse.json(
          { success: false, error: 'All boxes must have a valid slice index' },
          { status: 400 }
        );
      }
    }

    // Call the Python 3D segmentation script
    const segmentResult = await run3DSegmentation(volume, boxes, useMedical);

    if (segmentResult.error) {
      return NextResponse.json(
        { success: false, error: segmentResult.error },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      results: segmentResult.results,
    });
  } catch (error) {
    console.error('3D Segmentation error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

async function run3DSegmentation(
  volumeBase64: string,
  boxes: Box3D[],
  useMedical: boolean = false
): Promise<{ results?: Record<string, { WG?: string; CG?: string; PZ?: string }>; error?: string }> {
  const { spawn } = require('child_process');

  return new Promise((resolve, reject) => {
    const inputData = JSON.stringify({ volume: volumeBase64, boxes });

    console.log('[3D Segment] Input data size:', inputData.length);
    console.log('[3D Segment] Boxes:', JSON.stringify(boxes, null, 2));

    // Prepare environment variables
    const env = {
      ...process.env,
      USE_MEDICAL_SAM2: useMedical ? 'true' : 'false',
      PYTHONPATH: '/usr/local/lib/python3.12/dist-packages:/workspace/projects/Seg-code-try2region-noise',
    };

    // Use inference_3d.py for 3D volume segmentation
    const pythonProcess = spawn('/usr/bin/python3', ['scripts/inference_3d.py'], {
      env: env,
      cwd: '/workspace/projects',
    });

    console.log('[3D Segment] Python process started');

    // Write data to stdin
    pythonProcess.stdin.write(inputData);
    pythonProcess.stdin.end();

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data: Buffer) => {
      stdout += data.toString();
      console.log('[3D Segment] Python stdout chunk:', data.toString().substring(0, 100));
    });

    pythonProcess.stderr.on('data', (data: Buffer) => {
      stderr += data.toString();
      console.log('[3D Segment] Python stderr:', data.toString().substring(0, 200));
    });

    pythonProcess.on('close', (code: number) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdout);
          if (result.success) {
            resolve({ results: result.results });
          } else {
            resolve({ error: result.error || 'Segmentation failed' });
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
