
from PIL import Image
import os
import glob
import torch
from torchvision import transforms
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np
from torchvision.transforms.functional import rgb_to_grayscale

def load_esrgan_model(model_path: str, scale: int = 2, half: bool = True):
    """
    Load the RealESRGAN model.
    
    Parameters:
        model_path (str): Path to the pre-trained model.
        scale (int): The scaling factor (default is 2x).
        half (bool): Whether to use FP16 precision if supported (default is True).
    
    Returns:
        upsampler: The loaded RealESRGAN upsampler.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model architecture
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    
    # Initialize the Real-ESRGAN upscaler
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=half,  # Use FP16 inference if supported by your hardware
        device=device
    )
    
    return upsampler

def upscale_image(image: Image.Image, upsampler, outscale: int = 2) -> Image.Image:
    """
    Upscale an image using the RealESRGAN model.
    
    Parameters:
        image (PIL.Image.Image): The input image.
        upsampler: The RealESRGAN upsampler model.
        outscale (int): The output scale factor (default is 2x).
    
    Returns:
        PIL.Image.Image: The upscaled image.
    """
    # Convert PIL image to NumPy array
    img_np = np.array(image)
    
    # Upscale image
    output_img_np, _ = upsampler.enhance(img_np, outscale=outscale)
    
    # Convert NumPy array back to PIL image
    output_img = Image.fromarray(output_img_np)
    
    return output_img

def process_images(input_dir, output_dir):
    """
    Process images by cropping them to 1024x1024 and saving them to the output directory.

    Parameters:
    input_dir (str): The directory containing the input images.
    output_dir (str): The directory where the processed images will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the ESRGAN model
    model_path = "E:\Python Projects\FakeImageClass\Models\RealESRGAN_x2plus.pth"

    esrgan_model = load_esrgan_model(model_path, scale=2, half=True)
    
    # Iterate over all images in the input directory
    for image_path in glob.glob(os.path.join(input_dir, "*.png")):
        try:
            image = Image.open(image_path)
            
            # Check if the image is at least 1024x1024
            width, height = image.size

            # if both dimensions are larger than 1024, downscale the image while maintaining aspect ratio, until the smallest dimension is 1024
            if width > 1024 and height > 1024:
                print(f"Image {os.path.basename(image_path)} is larger than 1024x1024 and will be downscaled.")
                if width < height:
                    new_width = 1024
                    new_height = int(1024 * (height / width))
                    print('Width < Height')
                else:
                    new_height = 1024
                    new_width = int(1024 * (width / height))
                    print('Width > Height')
                image = image.resize((new_width, new_height), Image.LANCZOS)
                width, height = image.size
                print(f"Image {os.path.basename(image_path)} resized to {width}x{height}")

            # Upscale the image if it is smaller than 1024x1024 on either dimension
            if width < 1024 or height < 1024:
                print(f"Image {os.path.basename(image_path)} is smaller than 1024x1024 and will be upscaled.")
                image = upscale_image(image, esrgan_model, outscale=2)
                width, height = image.size
                print(f"Image {os.path.basename(image_path)} upscaled to {width}x{height}")
                if width < 1024 or height < 1024:
                    print(f"Upscaled image {os.path.basename(image_path)} is still smaller than 1024x1024 and will be skipped.")
                
            
            # Center crop the image to 1024x1024
            left = (width - 1024) / 2
            top = (height - 1024) / 2
            right = (width + 1024) / 2
            bottom = (height + 1024) / 2
            image = image.crop((left, top, right, bottom))

            # Save the image to the output directory
            image_name = os.path.basename(image_path)
            image.save(os.path.join(output_dir, image_name))

            print(f"Image {image_name} saved to {output_dir}")
        except Exception as e:
            print(f"Error processing image {os.path.basename(image_path)}: {e}")


input_dir = r"E:\Python Projects\FakeImageClass\Dataset\Original Dataset\DIV2K_valid_HR\DIV2K_valid_HR"
output_dir = r"E:\Python Projects\FakeImageClass\Dataset\Images\Valid"





process_images(input_dir, output_dir)
