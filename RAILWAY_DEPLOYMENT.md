# Railway Deployment Guide

## Prerequisites
- GitHub account with the ProZoneSAM2 repository
- Railway account (https://railway.app)

## Steps

### 1. Push code to GitHub
```bash
git add .
git commit -m "Add model download support"
git push origin master
```

### 2. Deploy on Railway
1. Go to https://railway.app
2. Login with GitHub
3. Click **"New Project"** → **"Deploy from GitHub repo"**
4. Select your **ProZoneSAM2** repository

### 3. Configure Environment Variables
In Railway project settings, add these variables:

| Variable | Value |
|----------|-------|
| `USE_PYTHON_SEGMENTATION` | `true` |
| `MEDICAL_MODEL_URL` | `https://huggingface.co/Selena1919/ProZoneSAM2/resolve/main/best_mean3d_model.pth` |
| `SAM_MODEL_URL` | `https://huggingface.co/Selena1919/ProZoneSAM2/resolve/main/sam2_hiera_small.pt` |

### 4. Railway will automatically:
1. Install Python 3.12
2. Install Node.js dependencies
3. Install PyTorch and other ML libraries
4. Download models from Hugging Face
5. Build and deploy the Next.js app

### 5. Wait for deployment
First deployment may take 5-10 minutes due to model downloads (~350MB total).

## Troubleshooting

### Python not found
Railway should install Python automatically. If not, add a `runtime.txt` file:
```
python-3.12
```

### Model download fails
Check that:
1. Hugging Face repository is public
2. URLs are correct (use `resolve/main`, not `blob/main`)

### Out of memory
SAM2 model requires ~2GB RAM. Use the small model variant if needed.
