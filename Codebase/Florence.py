import os
import pathlib
from glob import glob

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Florence2Processor, Florence2ForConditionalGeneration

# Get current directory
current_path = pathlib.Path(__file__).parent.absolute()
# Get parent directory
parent_path = current_path.parent.absolute()

input_dir = parent_path / "Dataset" / "Images" / "faces_dataset_small"
output_dir = parent_path / "Dataset" / "Captions" / "Faces"

# input_dir = parent_path / "Dataset" / "Original Dataset" / "faces_dataset_small_renamed" 
# output_dir = parent_path / "Dataset" / "Original Dataset" / "ffhq_captions"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Set the device to GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Set to FP16 if having a GPU
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model = AutoModelForCausalLM.from_pretrained(
#     "multimodalart/Florence-2-large-no-flash-attn",
#     dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#     trust_remote_code=True
# ).to(device)
# processor = AutoProcessor.from_pretrained(
#     "multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True, 
# )
MODEL_ID = "ducviet00/Florence-2-large-hf"
model = Florence2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True
).to(device)
processor = Florence2Processor.from_pretrained(
    MODEL_ID, trust_remote_code=True, 
)

prompt = "<DETAILED_CAPTION>"
existing_images = set(os.listdir(output_dir))

# Iterate over all images in the input directory
for image_path in tqdm(
    glob(os.path.join(input_dir, "*.png")), desc="Generating captions"
):
    image_name = os.path.basename(image_path)
    if f"{os.path.splitext(image_name)[0]}.txt" not in existing_images:
        image = Image.open(image_path)

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(
            device, torch_dtype
        )

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = processor.post_process_generation(
            generated_text, task="DETAILED_CAPTION", image_size=(image.width, image.height)
        )

        # Save the caption to a .txt file with the same name as the image
        image_name = os.path.basename(image_path)
        caption_path = output_dir / f"{os.path.splitext(image_name)[0]}.txt"

        cleaned_answer = parsed_answer["DETAILED_CAPTION"]
        with open(caption_path, "w") as f:
            f.write(cleaned_answer)
