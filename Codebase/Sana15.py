import os
import pathlib
import pickle
from glob import glob

# import cv2
import numpy as np
import torch
from diffusers import (
    BitsAndBytesConfig,
    SanaPipeline,
)

# from huggingface_hub import login
# from PIL import Image
from tqdm import tqdm

# login()

# Check if CUDA is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Get current directory
current_path = pathlib.Path(__file__).parent.absolute()
# Get parent directory
parent_path = current_path.parent.absolute()
input_dir = parent_path / "Dataset" /"Captions" / "Valid"
output_dir = parent_path / "Dataset" / "Images" / "Valid" / "sana15"
#validation_dir = parent_path / "Dataset" / "Images" / "Train" / "Generated"
os.makedirs(output_dir, exist_ok=True)

# check the output directory for existing images. If an image with the same name exists, skip generating it.
existing_images = set(os.listdir(output_dir))


pipeline = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16,
)
#pipeline.enable_model_cpu_offload()
#set to cuda
pipeline=pipeline.to(device)
pipeline.vae.to(torch.bfloat16)
pipeline.text_encoder.to(torch.bfloat16)


# Iterate over all txt files in the input directory
for filepath in tqdm(
    glob(os.path.join(input_dir, "*.txt")),
    desc="Generating images with Sana 1.5",
):
    image_filename = f"{os.path.basename(filepath)[:-4]}.png"
    #print(f"\n{image_filename}")

    if image_filename not in existing_images:



        caption = open(filepath).read()
        # print(f"Generating image for caption: {caption}")
        image = pipeline(
        prompt=caption,
        height=1024,
        width=1024,
        guidance_scale=4.5,
        num_inference_steps=20,
        generator=torch.Generator(device="cuda").manual_seed(1),
)[0]
        image[0].save(os.path.join(output_dir, image_filename))

        # # Convert image to numpy array
        # image_np = np.array(image)
        # # Step 2: Convert RGB to BGR for OpenCV
        # image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # # Step 3: Display the image using OpenCV
        # cv2.imshow("Image", image_bgr)
        # cv2.waitKey(0)  # Waits for a key press to close the window
        # cv2.destroyAllWindows()
