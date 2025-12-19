# Real-ESRGAN Video Upscaler - RunPod Serverless Worker

RunPod serverless worker for upscaling videos using Real-ESRGAN. Converts 720p (1280x720) videos to 1080p (1920x1080) with AI-powered super-resolution.

## Features

- **Video Upscaling**: 720p → 1080p (1.5x scale)
- **GPU Accelerated**: Uses CUDA for fast processing
- **Memory Efficient**: Tiling support for large videos
- **Base64 I/O**: Accept and return base64-encoded videos
- **Automatic Model Loading**: Pre-downloads model weights in Docker image

## Quick Start

### 1. Clone or Use Repository

```bash
git clone git@github.com:Th3CracKed/Real-ESRGAN.git
cd Real-ESRGAN
```

### 2. Test Locally (Optional)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model weights manually if not using Docker
python -c "from basicsr.utils.download_util import load_file_from_url; \
    load_file_from_url(\
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', \
        model_dir='weights', \
        progress=True\
    )"

# Test with sample input
python handler.py
```

### 3. Deploy to RunPod

#### Option A: GitHub Integration (Recommended)

1. Push this repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Real-ESRGAN video upscaler"
   git remote add origin git@github.com:Th3CracKed/Real-ESRGAN.git
   git push -u origin main
   ```

2. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)

3. Click **"New Template"** → **"GitHub"**

4. Connect your GitHub account and select the `Th3CracKed/Real-ESRGAN` repository

5. RunPod will automatically build and deploy your worker

#### Option B: Manual Docker Build

```bash
# Build Docker image
docker build -t your-dockerhub-username/realesrgan-video-upscaler:latest .

# Push to Docker Hub
docker push your-dockerhub-username/realesrgan-video-upscaler:latest

# Create endpoint in RunPod UI pointing to your image
```

## API Usage

### Input Format

```json
{
  "input": {
    "video": "<base64_encoded_mp4_video>",
    "scale": 1.5
  }
}
```

**Parameters:**
- `video` (required): Base64-encoded MP4 video
- `scale` (optional): Output scale multiplier (default: 1.5 for 720p→1080p)

### Output Format

```json
{
  "video": "<base64_encoded_upscaled_video>",
  "input_resolution": "1280x720",
  "output_resolution": "1920x1080",
  "frame_count": 150,
  "processing_time": 45.2
}
```

### Error Response

```json
{
  "error": "Error message here",
  "processing_time": 12.3
}
```

## Integration Example (Node.js/TypeScript)

```typescript
import fetch from 'node-fetch';
import * as fs from 'fs';

async function upscaleVideo(videoPath: string): Promise<string> {
  // Read and encode video
  const videoBuffer = fs.readFileSync(videoPath);
  const videoBase64 = videoBuffer.toString('base64');
  
  // Call RunPod endpoint
  const response = await fetch('https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.RUNPOD_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      input: {
        video: videoBase64,
        scale: 1.5
      }
    })
  });
  
  const result = await response.json();
  const jobId = result.id;
  
  // Poll for completion
  let output;
  while (true) {
    const statusResponse = await fetch(
      `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/${jobId}`,
      {
        headers: {
          'Authorization': `Bearer ${process.env.RUNPOD_API_KEY}`
        }
      }
    );
    
    const status = await statusResponse.json();
    
    if (status.status === 'COMPLETED') {
      output = status.output;
      break;
    } else if (status.status === 'FAILED') {
      throw new Error(`Job failed: ${status.error}`);
    }
    
    // Wait 5 seconds before next poll
    await new Promise(resolve => setTimeout(resolve, 5000));
  }
  
  // Decode and save upscaled video
  const upscaledBuffer = Buffer.from(output.video, 'base64');
  fs.writeFileSync('upscaled_video.mp4', upscaledBuffer);
  
  console.log(`Upscaled: ${output.input_resolution} → ${output.output_resolution}`);
  console.log(`Processing time: ${output.processing_time}s`);
  
  return output.video; // Return base64 string
}
```

## Configuration

### Environment Variables

- `MODEL_NAME`: Model to use (default: `RealESRGAN_x4plus`)
  - `RealESRGAN_x4plus`: General purpose (recommended)
  - `realesr-animevideov3`: Optimized for anime videos

### GPU Requirements

- **Minimum**: NVIDIA GPU with 6GB VRAM
- **Recommended**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **Tile Size**: Adjust in `handler.py` if you encounter OOM errors

### Processing Times

Approximate times for a 5-second 720p video (150 frames):

| GPU | Processing Time |
|-----|----------------|
| RTX 3090 | ~30-45 seconds |
| RTX 3060 | ~60-90 seconds |
| T4 (RunPod) | ~90-120 seconds |

## Troubleshooting

### CUDA Out of Memory

Reduce tile size in `handler.py`:

```python
UPSAMPLER = RealESRGANer(
    # ... other params ...
    tile=128,  # Reduce from 256 to 128 or 64
)
```

### ffmpeg Not Found

Ensure Docker image includes ffmpeg (already configured in Dockerfile).

### Model Not Loading

Check that model weights are properly downloaded:

```bash
docker run your-image ls -lh /weights/
```

Should show:
```
-rw-r--r-- 1 root root 65M RealESRGAN_x4plus.pth
```

## Repository Structure

```
Real-ESRGAN/
├── handler.py           # Main RunPod handler
├── Dockerfile          # Docker image configuration
├── requirements.txt    # Python dependencies
├── test_input.json    # Sample input for testing
└── README.md          # This file
```

## Model Information

This worker uses [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al.

**Citation:**
```bibtex
@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```

## License

MIT License - See original [Real-ESRGAN repository](https://github.com/xinntao/Real-ESRGAN) for model license.

## Support

For issues or questions:
- RunPod Worker Issues: [Create an issue](https://github.com/Th3CracKed/Real-ESRGAN/issues)
- Real-ESRGAN Model: [Original repository](https://github.com/xinntao/Real-ESRGAN)
- RunPod Documentation: [docs.runpod.io](https://docs.runpod.io)
