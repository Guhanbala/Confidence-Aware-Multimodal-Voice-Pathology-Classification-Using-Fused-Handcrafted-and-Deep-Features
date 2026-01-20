# 🛠️ Setup Guide

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/voice-pathology-detection.git
cd voice-pathology-detection
```

### 2. Create Virtual Environment

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install FFmpeg

FFmpeg is required for audio file conversion.

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

#### macOS (with Homebrew):
```bash
brew install ffmpeg
```

#### Windows:
1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add to PATH:
   - Search "Environment Variables" in Windows
   - Edit "Path" under System Variables
   - Add `C:\ffmpeg\bin`
4. Verify: `ffmpeg -version`

### 5. Prepare Your Dataset

Create a `dataset/` folder with the following structure:

```
dataset/
├── healthy/
│   ├── overview.csv
│   └── [audio files]
├── Laryngitis/
│   ├── overview.csv
│   └── [audio files]
├── Hyperfunktionelle Dysphonie/
│   ├── overview.csv
│   └── [audio files]
├── Kontaktpachydermie/
│   ├── overview.csv
│   └── [audio files]
└── Rekurrensparese/
    ├── overview.csv
    └── [audio files]
```

**Important**: Each `overview.csv` must contain:
- `AufnahmeID` - Patient ID
- `Geburtsdatum` - Birth date (YYYY-MM-DD format)
- `AufnahmeDatum` - Recording date (YYYY-MM-DD format)

### 6. Verify Installation

Test your setup:

```bash
python -c "import torch; import librosa; import cv2; print('✅ All dependencies installed!')"
```

### 7. Run the Pipeline

Execute scripts in order:

```bash
# Step 1: Prepare data
python data.py

# Step 2: Extract features
python feature_extraction.py

# Step 3: Extract deep features (requires GPU for faster training)
python resnet18_deep_features.py

# Step 4: Fuse features
python fuse_data.py

# Step 5: Train and evaluate
python final.py
```

## 🐛 Common Issues

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Make sure virtual environment is activated and requirements are installed:
```bash
pip install -r requirements.txt
```

### Issue: "FFmpeg not found"
**Solution**: 
- Verify installation: `ffmpeg -version`
- Ensure FFmpeg is in your system PATH

### Issue: "CUDA out of memory"
**Solution**: 
- Edit `resnet18_deep_features.py` and reduce `BATCH_SIZE`
- Or use CPU: change `DEVICE = "cpu"`

### Issue: "No data processed" when running data.py
**Solution**:
- Check that `dataset/` folder exists with correct structure
- Verify `overview.csv` files are present and properly formatted
- Check audio file naming convention (should contain "a_n" and patient ID)

### Issue: Virtual environment activation not working
**Windows PowerShell**: If you get an execution policy error:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 💡 Tips

- **GPU Acceleration**: For faster training, use a CUDA-capable GPU
- **Disk Space**: Ensure you have at least 5GB free for generated features
- **Python Version**: Tested on Python 3.8-3.11
- **RAM**: Minimum 8GB recommended

## 📦 Deactivating Virtual Environment

When you're done:
```bash
deactivate
```

## 🔄 Updating Dependencies

To update all packages to their latest versions:
```bash
pip install --upgrade -r requirements.txt
```

## 🆘 Getting Help

If you encounter issues:
1. Check the [Troubleshooting](#-common-issues) section
2. Review error messages carefully
3. Open an issue on GitHub with:
   - Error message
   - Python version (`python --version`)
   - OS version
   - Steps to reproduce
