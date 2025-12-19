"""
RunPod Serverless Worker for Real-ESRGAN Video Upscaling
Upscales videos from 720p (1280x720) to 1080p (1920x1080) using Real-ESRGAN
"""

import runpod
import torch
import cv2
import numpy as np
import base64
import tempfile
import os
import subprocess
from pathlib import Path
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# Global model initialization (loads once per worker instance)
MODEL = None
UPSAMPLER = None

def initialize_model():
    """Initialize the Real-ESRGAN model (called once per worker)"""
    global MODEL, UPSAMPLER
    
    if UPSAMPLER is not None:
        return
    
    print("[Real-ESRGAN] Initializing model...")
    
    # Use RealESRGAN_x4plus for general video upscaling
    # For anime videos, use realesr-animevideov3 instead
    model_name = os.getenv('MODEL_NAME', 'RealESRGAN_x4plus')
    
    if model_name == 'realesr-animevideov3':
        MODEL = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, 
            num_conv=16, upscale=4, act_type='prelu'
        )
        model_path = 'weights/realesr-animevideov3.pth'
        netscale = 4
    else:  # RealESRGAN_x4plus (default)
        MODEL = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, 
            num_block=23, num_grow_ch=32, scale=4
        )
        model_path = 'weights/RealESRGAN_x4plus.pth'
        netscale = 4
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please download the model weights to the weights/ directory."
        )
    
    # Initialize upsampler with GPU support
    half = True if torch.cuda.is_available() else False
    UPSAMPLER = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=MODEL,
        tile=256,  # Use tiling to handle large videos without OOM
        tile_pad=10,
        pre_pad=0,
        half=half,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    print(f"[Real-ESRGAN] Model initialized: {model_name}")
    print(f"[Real-ESRGAN] Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"[Real-ESRGAN] Half precision: {half}")


def decode_video_from_base64(video_base64):
    """Decode base64 video to temporary file"""
    video_data = base64.b64decode(video_base64)
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_input.write(video_data)
    temp_input.close()
    return temp_input.name


def encode_video_to_base64(video_path):
    """Encode video file to base64"""
    with open(video_path, 'rb') as video_file:
        video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
    return video_base64


def get_video_info(video_path):
    """Get video metadata (fps, frame count, dimensions)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    return fps, frame_count, width, height


def upscale_video(input_path, output_path, scale=1.5):
    """
    Upscale video using Real-ESRGAN
    Args:
        input_path: Path to input video
        output_path: Path to save upscaled video
        scale: Output scale (1.5 for 720p->1080p: 1280x720 * 1.5 = 1920x1080)
    """
    print(f"[Real-ESRGAN] Processing video: {input_path}")
    
    # Get video info
    fps, frame_count, input_width, input_height = get_video_info(input_path)
    print(f"[Real-ESRGAN] Input: {input_width}x{input_height}, {frame_count} frames @ {fps} fps")
    
    # Calculate output dimensions
    output_width = int(input_width * scale)
    output_height = int(input_height * scale)
    print(f"[Real-ESRGAN] Output: {output_width}x{output_height}")
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    temp_output_frames = os.path.join(temp_dir, 'upscaled_%05d.png')
    
    try:
        # Process frames
        frame_idx = 0
        print(f"[Real-ESRGAN] Upscaling {frame_count} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Upscale frame using Real-ESRGAN
            try:
                output_frame, _ = UPSAMPLER.enhance(frame, outscale=scale)
            except RuntimeError as error:
                print(f"[Real-ESRGAN] Error upscaling frame {frame_idx}: {error}")
                print("[Real-ESRGAN] Hint: If CUDA OOM, reduce tile size or use CPU")
                raise
            
            # Save upscaled frame
            frame_path = temp_output_frames % frame_idx
            cv2.imwrite(frame_path, output_frame)
            
            frame_idx += 1
            if frame_idx % 30 == 0:  # Progress every 30 frames
                progress = (frame_idx / frame_count) * 100
                print(f"[Real-ESRGAN] Progress: {frame_idx}/{frame_count} ({progress:.1f}%)")
        
        cap.release()
        print(f"[Real-ESRGAN] Upscaled {frame_idx} frames")
        
        # Encode frames back to video using ffmpeg
        print("[Real-ESRGAN] Encoding video with ffmpeg...")
        input_pattern = os.path.join(temp_dir, 'upscaled_%05d.png')
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # High quality
            '-y',  # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        
        print(f"[Real-ESRGAN] Video saved: {output_path}")
        
        # Get output file size
        output_size = os.path.getsize(output_path)
        print(f"[Real-ESRGAN] Output size: {output_size / 1024 / 1024:.2f} MB")
        
    finally:
        # Cleanup temporary frames
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def handler(job):
    """
    RunPod handler function for video upscaling
    
    Expected input format:
    {
        "input": {
            "video": "<base64_encoded_video>",  # Base64 encoded MP4 video
            "scale": 1.5,  # Optional: output scale (default: 1.5 for 720p->1080p)
            "model": "RealESRGAN_x4plus"  # Optional: model name
        }
    }
    
    Returns:
    {
        "video": "<base64_encoded_upscaled_video>",
        "input_resolution": "1280x720",
        "output_resolution": "1920x1080",
        "frame_count": 150,
        "processing_time": 45.2
    }
    """
    import time
    start_time = time.time()
    
    try:
        # Initialize model (first call only)
        initialize_model()
        
        # Get input
        job_input = job["input"]
        
        # Validate input
        if "video" not in job_input:
            return {
                "error": "Missing 'video' field in input. Expected base64 encoded video."
            }
        
        video_base64 = job_input["video"]
        scale = float(job_input.get("scale", 1.5))  # Default 1.5x for 720p->1080p
        
        print(f"[Real-ESRGAN] Received upscaling request (scale: {scale}x)")
        
        # Decode input video
        input_path = decode_video_from_base64(video_base64)
        print(f"[Real-ESRGAN] Decoded input video: {input_path}")
        
        # Get input resolution
        _, frame_count, input_width, input_height = get_video_info(input_path)
        input_resolution = f"{input_width}x{input_height}"
        
        # Create output path
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='_upscaled.mp4').name
        
        try:
            # Upscale video
            upscale_video(input_path, output_path, scale=scale)
            
            # Get output resolution
            _, _, output_width, output_height = get_video_info(output_path)
            output_resolution = f"{output_width}x{output_height}"
            
            # Encode output video to base64
            video_output_base64 = encode_video_to_base64(output_path)
            
            processing_time = time.time() - start_time
            
            print(f"[Real-ESRGAN] Success! {input_resolution} -> {output_resolution} in {processing_time:.1f}s")
            
            return {
                "video": video_output_base64,
                "input_resolution": input_resolution,
                "output_resolution": output_resolution,
                "frame_count": frame_count,
                "processing_time": round(processing_time, 2)
            }
        
        finally:
            # Cleanup temporary files
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Error during upscaling: {str(e)}"
        print(f"[Real-ESRGAN] {error_msg}")
        
        return {
            "error": error_msg,
            "processing_time": round(processing_time, 2)
        }


# Start the RunPod serverless worker
if __name__ == "__main__":
    print("[Real-ESRGAN] Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
