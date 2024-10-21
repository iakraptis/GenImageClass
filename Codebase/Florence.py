import os
import glob
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Define input and output directories
input_dir = r"E:\Python Projects\FakeImageClass\Dataset\Images"
output_dir = r"E:\Python Projects\FakeImageClass\Dataset\Captions"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)

prompt = "<DETAILED_CAPTION>"

# Iterate over all images in the input directory
for image_path in glob.glob(os.path.join(input_dir, "*.jpg")):
    image = Image.open(image_path)
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    parsed_answer = processor.post_process_generation(generated_text, task="DETAILED_CAPTION", image_size=(image.width, image.height))
    
    # Save the caption to a .txt file with the same name as the image
    image_name = os.path.basename(image_path)
    caption_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")

    cleaned_answer = parsed_answer['DETAILED_CAPTION']
    with open(caption_path, "w") as f:
        f.write(cleaned_answer)

    print(f"Caption for {image_name} saved to {caption_path}")