# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import tempfile
from pathlib import Path
import time
import torch
from cog import BasePredictor, Input, Path
from utils import utils_image as util
from models.network_rrdbnet import RRDBNet as net
from tqdm import tqdm

WEIGHTS_BASE_URL = f"https://weights.replicate.delivery/default/BSRGAN"
MODEL_FILES = [
    "model_zoo.tar",
]


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)

    if ".tar" in dest:
        dest = dest.replace(".tar", "")

    command = ["pget", "-vfx", url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self):
        # Add download logic at the start of setup
        for model_file in MODEL_FILES:
            url = WEIGHTS_BASE_URL + "/" + model_file
            dest_path = model_file

            dir_name = dest_path.replace(".tar", "")
            print(f"dir_name: {dir_name}")
            if os.path.exists(dir_name):
                print(f"[+] Directory {dir_name} already exists, skipping download")
                continue

            download_weights(url, dest_path)

        # Initialize models dictionary for lazy loading
        self.models = {
            2: {"name": "BSRGANx2", "model": None},
            4: {"name": "BSRGAN", "model": None},
        }

    def load_model(self, scale_factor):
        """Load the appropriate model for the scale factor"""
        if self.models[scale_factor]["model"] is None:
            model_path = f"model_zoo/{self.models[scale_factor]['name']}.pth"

            # Initialize model with correct scale factor
            model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=scale_factor)
            model.load_state_dict(
                torch.load(model_path, weights_only=True), strict=True
            )
            model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()

            self.models[scale_factor]["model"] = model

        return self.models[scale_factor]["model"]

    def process_image(self, model, input_path, output_path):
        """Process a single image through the model"""
        # Read and prepare image tensor
        img = util.imread_uint(str(input_path), n_channels=3)
        img_tensor = util.uint2tensor4(img)

        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        # Run inference
        with torch.no_grad():
            output = model(img_tensor)

        # Convert back to image and save
        result = util.tensor2uint(output)
        util.imsave(result, str(output_path))

    def process_video(self, model, input_path, output_path):
        """Process video by extracting frames, upscaling each with process_image, and recombining"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            frame_dir = tmp / "frames"
            frame_dir.mkdir()
            processed_dir = tmp / "processed"
            processed_dir.mkdir()

            # Get input video's frame rate
            fps_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(input_path)
            ]
            fps = subprocess.check_output(fps_cmd).decode().strip()
            # Convert fraction (e.g., "30000/1001") to float
            if '/' in fps:
                num, den = map(int, fps.split('/'))
                fps = str(round(num / den, 2))
            print(f"Input video frame rate: {fps} fps")

            # Extract all frames to disk using source fps
            subprocess.run([
                "ffmpeg", "-i", str(input_path),
                "-vf", f"fps={fps}",
                str(frame_dir / "frame_%04d.png")
            ], check=True)

            # Process each frame using our fast process_image function
            frames = sorted(frame_dir.glob("*.png"))
            for frame_path in tqdm(frames, desc="Processing frames"):
                self.process_image(model, frame_path, processed_dir / frame_path.name)

            # Recombine into video using source fps
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", fps,
                "-i", str(processed_dir / "frame_%04d.png"),
                "-i", str(input_path),  # Original for audio
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                str(output_path)
            ], check=True)

    def predict(
        self,
        input_file: Path = Input(description="Input image or video file"),
        scale_factor: int = Input(
            description="Upscaling factor (2x or 4x)", choices=[2, 4], default=4
        ),
    ) -> Path:
        """Main prediction endpoint"""
        # Validate scale factor
        if scale_factor not in [2, 4]:
            raise ValueError("Scale factor must be 2 or 4")

        # Load appropriate model
        model = self.load_model(scale_factor)

        # Determine output path
        input_path = Path(input_file)
        is_image = input_path.suffix.lower() in [".png", ".jpg", ".jpeg"]
        output_path = Path("output.png" if is_image else "output.mp4")

        # Process based on file type
        if is_image:
            self.process_image(model, input_path, output_path)
        else:
            self.process_video(model, input_path, output_path)

        return output_path
