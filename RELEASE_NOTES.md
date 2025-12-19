# Real-ESRGAN Video Upscaler v1.0.0

## 🎉 Initial Release

First stable release of the Real-ESRGAN Video Upscaler for RunPod Serverless.

### ✨ Features

- **AI-Powered Video Upscaling**: Upscale 720p videos to 1080p using Real-ESRGAN
- **GPU Accelerated**: Optimized for NVIDIA GPUs with CUDA support
- **Memory Efficient**: Tiling support for processing large videos without OOM errors
- **Base64 I/O**: Simple API with base64-encoded video input/output
- **Automatic Model Loading**: Pre-downloads Real-ESRGAN weights in Docker image
- **Frame-by-Frame Processing**: High-quality super-resolution for every frame
- **Progress Tracking**: Real-time progress updates during processing

### 📦 What's Included

- `handler.py`: Main RunPod serverless handler
- `Dockerfile`: Optimized Docker image with CUDA 11.8
- `requirements.txt`: All necessary Python dependencies
- `.runpod/hub.json`: RunPod Hub configuration
- `.runpod/tests.json`: Automated testing configuration
- Complete documentation and integration examples

### 🚀 Deployment

Deploy using RunPod's GitHub integration:
1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click "New Template" → "GitHub"
3. Select `Th3CracKed/Real-ESRGAN` repository
4. RunPod will automatically build and deploy

Or use the RunPod Hub:
[![RunPod](https://api.runpod.io/badge/Th3CracKed/Real-ESRGAN)](https://console.runpod.io/hub/Th3CracKed/Real-ESRGAN)

### 📖 API Usage

```json
{
  "input": {
    "video": "<base64_encoded_video>",
    "scale": 1.5
  }
}
```

**Response:**
```json
{
  "video": "<base64_encoded_upscaled_video>",
  "input_resolution": "1280x720",
  "output_resolution": "1920x1080",
  "frame_count": 150,
  "processing_time": 45.2
}
```

### ⚙️ Configuration

- **MODEL_NAME**: Choose between `RealESRGAN_x4plus` (general) or `realesr-animevideov3` (anime)
- **GPU**: Requires NVIDIA GPU with 6GB+ VRAM
- **Processing Time**: ~30-120 seconds for a 5-second video (depending on GPU)

### 🔧 Requirements

- **GPU**: NVIDIA GPU with CUDA support (6GB+ VRAM recommended)
- **Docker**: For local testing and manual deployment
- **Python 3.11**: If running without Docker

### 📝 Credits

This project uses [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al.

### 🐛 Known Issues

None at this time.

### 📄 License

MIT License - See LICENSE file for details.

---

**Full Changelog**: https://github.com/Th3CracKed/Real-ESRGAN/commits/v1.0.0
