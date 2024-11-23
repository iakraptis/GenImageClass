import json
import os
import pathlib
from glob import glob

import requests
from dotenv import load_dotenv
from openai import AzureOpenAI
from PIL import Image
from tqdm import tqdm

# Load the environment variables
load_dotenv(override=True)
DEPLOYMENT_NAME = os.getenv("DALLE_DEPLOYMENT_NAME")
client = AzureOpenAI(
    api_version="2024-02-01",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Get current directory
current_path = pathlib.Path(__file__).parent.absolute()
# Get parent directory
parent_path = current_path.parent.absolute()
input_dir = parent_path / "Dataset" / "Captions" / "Faces"
output_dir = parent_path / "Dataset" / "Images" / "Train" / "Generated" / "Faces_Dalle"
validation_dir = (
    parent_path / "Dataset" / "Images" / "Valid" / "Generated" / "Faces_Dalle"
)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
generated_images = os.listdir(output_dir) + os.listdir(validation_dir)


for filepath in tqdm(
    glob(os.path.join(input_dir, "*.txt")),
    desc="Generating images with Dalle 3 from Azure Deployment",
):
    image_filename = f"{os.path.basename(filepath)[:-4]}.png"
    if image_filename not in generated_images:
        caption = open(filepath).read()
        result = client.images.generate(
            model=DEPLOYMENT_NAME,
            prompt=caption,
            n=1,
        )
        json_response = json.loads(result.model_dump_json())

        # Retrieve the generated image
        image_url = json_response["data"][0]["url"]  # extract image URL from response
        generated_image = requests.get(image_url).content  # download the image
        image_path = os.path.join(output_dir, image_filename)
        with open(image_path, "wb") as image_file:
            image_file.write(generated_image)
