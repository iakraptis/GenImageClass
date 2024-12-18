import os
import pathlib
import pickle
from glob import glob

# import cv2
import numpy as np
import torch
from diffusers import (
    BitsAndBytesConfig,
    SD3Transformer2DModel,
    StableDiffusionPipeline,
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
input_dir = parent_path / "Dataset" / "Captions" / "Valid"
output_dir = parent_path / "Dataset" / "Images" / "Valid" / "sd15"
#validation_dir = parent_path / "Dataset" / "Images" / "Train" / "Generated"
os.makedirs(output_dir, exist_ok=True)

# Read pickle file
# with open(current_path / "generated_images.pkl", "rb") as f:
#     generated_images = pickle.load(f)
# generated_images = os.listdir(output_dir) + os.listdir(validation_dir)
# # save to pickle
# with open(current_path / "generated_images.pkl", "wb") as f:
#     pickle.dump(generated_images, f)
# breakpoint()
#print (generated_images)
#breakpoint()
model_id = "sd-legacy/stable-diffusion-v1-5"


pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,  torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()
#set to cuda
#pipeline=pipeline.to(device)

# Iterate over all images in the input directory
for filepath in tqdm(
    glob(os.path.join(input_dir, "*.txt")),
    desc="Generating images with Stable Diffusion 1.5",
):
    image_filename = f"{os.path.basename(filepath)[:-4]}.png"
    print(image_filename)

    if image_filename :
        caption = open(filepath).read()
        # print(f"Generating image for caption: {caption}")
        image = pipeline(caption).images[0]
        image.save(os.path.join(output_dir, image_filename))

        # # Convert image to numpy array
        # image_np = np.array(image)
        # # Step 2: Convert RGB to BGR for OpenCV
        # image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # # Step 3: Display the image using OpenCV
        # cv2.imshow("Image", image_bgr)
        # cv2.waitKey(0)  # Waits for a key press to close the window
        # cv2.destroyAllWindows()
