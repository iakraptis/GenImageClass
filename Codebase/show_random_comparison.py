import os
import pathlib
import random
from glob import glob

import cv2
import matplotlib.pyplot as plt

current_path = pathlib.Path(__file__).parent.parent.absolute()
generated_images_dir = current_path / "Dataset" / "Images" / "Train" / "Generated"
actual_images_dir = current_path / "Dataset" / "Images" / "Train" / "Actual"
print(generated_images_dir)
print(actual_images_dir)

generated_images_paths = glob(str(generated_images_dir / "*.png"))
generated_images_paths = sorted(generated_images_paths)

random_image_path = random.choice(generated_images_paths)
random_image = cv2.imread(random_image_path)
random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)
os.path.basename(random_image_path)
actual_image_path = actual_images_dir / os.path.basename(random_image_path)
actual_image = cv2.imread(str(actual_image_path))
actual_image = cv2.cvtColor(actual_image, cv2.COLOR_BGR2RGB)

# peak a number 0 or 1
random_number = random.randint(0, 1)
plt.figure(figsize=(10, 10))
if random_number == 0:
    # show the images
    plt.subplot(1, 2, 1)
    plt.imshow(random_image)
    plt.title("Generated Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(actual_image)
    plt.title("Actual Image")
else:
    # show the images
    plt.subplot(1, 2, 1)
    plt.imshow(actual_image)
    plt.title("Actual Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(random_image)
    plt.title("Generated Image")

plt.axis("off")
plt.tight_layout()
plt.show()
