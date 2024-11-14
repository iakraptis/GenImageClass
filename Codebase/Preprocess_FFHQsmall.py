import os
import pathlib
from glob import glob

from PIL import Image

# This function iterates through the images in the input directory, renames them starting from 901, and saves them in the output directory
def rename_images(input_dir, output_dir):
    # Get all images in the input directory
    img_paths = sorted(glob(os.path.join(input_dir, '*')))

    # Iterate through the images
    for idx, img_path in enumerate(img_paths):
        try:
            # Load the image
            img = Image.open(img_path)
            # Save the image with a new name
            img.save(os.path.join(output_dir, f'{idx + 901}.png'))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    # Get current directory
    current_path = pathlib.Path(__file__).parent.parent.absolute()
    # Define the input directory
    input_dir = (
        current_path
        / "Dataset"
        / "Original Dataset"
        / "faces_dataset_small"
    )
    #breakpoint()
    # Define the output directory
    output_dir = (
        current_path
        / "Dataset"
        / "Original Dataset"
        / "faces_dataset_small_renamed"
    )
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Rename the images
    rename_images(input_dir, output_dir)
    print("Images renamed successfully.")