# GenImageClass
This project aim is to train classifier in order to distinguish if an image is real or ai generated. We use an **Image -> Caption -> Image** proccess and then we use this dataset in order to train various classifiers to distinguish the fake ones!

## TODO List:
- [ ] Investigate more suitable models
- [ ] Add Vscode for standarization (formaters)?

  

# Preprocess.py

This script iterates over a dataset an image dataset and standarizes every image to be 1024*1024. It achieves this by either upscaling an image or cropping it.

# Florence.py
The florence script is used to parse Actual Images ang generate captions for each one in order to be used later on the image creation process.

# Stable Diffusion.py
These scripts use Stable Diffusion 3.5, Stable Diffusion 1.5 and Sana 1.5 in order to generate "fake" images from the captions similar to the actual ones.

## Examples
![App Screenshot](./asset/Figure_1.png)
![App Screenshot](./asset/Figure_2.png)

# Folder Structure

```
asset
Codebase
Models
Dataset
├── Captions
│   ├── Train
|   └── Valid
├── Images
|   |
│   ├── Train
|   |   ├── Actual
|   |   ├── sd35
|   |   ├── sd15
|   |   └── sana15
│   └── Valid
|       ├── Actual
|       ├── sd35
|       ├── sd15
|       └── sana15
|
└── Original Dataset
    ├── DIV2K_train_HR
    │   └── DIV2K_train_HR
    └── DIV2K_valid_HR
        └── DIV2K_valid_HR
```
