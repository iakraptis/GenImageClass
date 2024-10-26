import os
import pathlib
from glob import glob

import cv2
import numpy as np
import torch
from diffusers import (
    BitsAndBytesConfig,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from PIL import Image
from tqdm import tqdm

# Get current directory
current_path = pathlib.Path(__file__).parent.absolute()
input_dir = current_path / "Dataset" / "Captions" / "Train"
output_dir = current_path / "Dataset" / "Images" / "Train" / "Generated"
os.makedirs(output_dir, exist_ok=True)

generated_images = os.listdir(output_dir)

model_id = "stabilityai/stable-diffusion-3.5-large"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
# Have a .env file with HF_TOKEN containing the API key created from your Hugging Face account
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16,
    token=os.getenv("HF_TOKEN"),
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, transformer=model_nf4, torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

# Iterate over all images in the input directory
for filepath in tqdm(
    glob(os.path.join(input_dir, "*.txt")),
    desc="Generating images with Stable Diffusion 3.5",
):
    image_filename = f"{os.path.basename(filepath)[:-4]}.png"
    if image_filename not in generated_images:
        caption = open(filepath).read()
        image = pipeline(caption, num_inference_steps=30, guidance_scale=3.5).images[0]
        image.save(os.path.join(output_dir, image_filename))

        # # Convert image to numpy array
        # image_np = np.array(image)
        # # Step 2: Convert RGB to BGR for OpenCV
        # image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # # Step 3: Display the image using OpenCV
        # cv2.imshow("Image", image_bgr)
        # cv2.waitKey(0)  # Waits for a key press to close the window
        # cv2.destroyAllWindows()
