#!/bin/bash
# Medical SAM2 Python Environment Setup Script

echo "=========================================="
echo "Setting up Python Environment for Medical SAM2"
echo "=========================================="

# Install system dependencies
echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1

# Install core ML libraries
echo "Installing PyTorch..."
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q

# Install Medical Imaging libraries
echo "Installing medical imaging libraries..."
python3 -m pip install monai nibabel SimpleITK pydicom -q

# Install Scientific Computing libraries
echo "Installing scientific computing libraries..."
python3 -m pip install numpy scipy scikit-image scikit-learn -q

# Install Image Processing libraries
echo "Installing image processing libraries..."
python3 -m pip install opencv-python matplotlib -q

# Install Configuration libraries
echo "Installing configuration libraries..."
python3 -m pip install hydra-core==1.3.2 omegaconf==2.3.0 -q

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python3 << 'EOF'
import torch, torchvision, monai, nibabel, scipy, cv2
print('✓ torch:', torch.__version__)
print('✓ torchvision:', torchvision.__version__)
print('✓ monai:', monai.__version__)
print('✓ nibabel:', nibabel.__version__)
print('✓ scipy:', scipy.__version__)
print('✓ opencv:', cv2.__version__)
from hydra import initialize_config_module
print('✓ hydra: function imported successfully')
print('')
print('All dependencies installed successfully!')
EOF

echo ""
echo "=========================================="
echo "Python Environment Information"
echo "=========================================="
python3 --version
python3 -c "import sys; print('Package directory:', '/usr/local/lib/python3.12/dist-packages')"

echo ""
echo "Environment setup completed!"
echo ""
echo "Next.js API will use these environment variables:"
echo "  PYTHONPATH=/usr/local/lib/python3.12/dist-packages:/workspace/projects/Seg-code-try2region-noise"
echo ""
