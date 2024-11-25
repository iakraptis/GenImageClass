import pathlib
import torch
from torchvision import transforms
from PIL import Image
from MultiClassifier import FakeImageClassifier, inference_transform

def load_model(model_path, device):
    model = FakeImageClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = inference_transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        rounded_outputs = torch.round(outputs, decimals=2)

        print(rounded_outputs)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return predicted_class

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model_path = "triplemodel.pth"
    model = load_model(model_path, device)

    # Path to the image to be predicted
    image_path = r"asset\0801sd35.png"

    # Predict the class of the image
    predicted_class = predict_image(model, image_path, device)
    # create a dictionary to map the class index to the class name
    class_map = {0: "Real", 1: "SD1.5", 2: "SD3.5"}
    predicted_class = class_map[predicted_class]

    print(f"Predicted class: {predicted_class}")