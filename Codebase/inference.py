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

    # Make prediction and convert logits to probabilities
    with torch.no_grad():
        outputs = model(image)  # raw logits
        probs = torch.softmax(outputs, dim=1)
        probs_np = probs.cpu().numpy()[0]

        # Class names must match your training class ordering
        class_names = ["Real", "Sana 1.5", "SD 1.5", "SD 3.5"]

        # Print per-class probabilities as percentages
        print("Prediction probabilities:")
        for idx, name in enumerate(class_names):
            print(f"  {name}: {probs_np[idx]*100:.2f}%")

        predicted_class = int(probs.argmax(dim=1).item())
        predicted_percent = probs_np[predicted_class] * 100
        print(f"Predicted class: {class_names[predicted_class]} ({predicted_percent:.2f}%)")

    # Return predicted index and probability vector (numpy)
    return predicted_class, probs_np

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model_path = "quadmodel.pth"
    model = load_model(model_path, device)

    # Path to the image to be predicted
    image_path = r"asset\0801dl3.jpg"

    # Predict the class of the image
    predicted_class_idx, probs = predict_image(model, image_path, device)
    # create a dictionary to map the class index to the class name
    class_map = {0: "Real", 1: "Sana 1.5", 2: "SD 1.5", 3: "SD 3.5"}
    predicted_class_name = class_map[predicted_class_idx]

    # Print a concise final line
    print(f"Final prediction: {predicted_class_name} ({probs[predicted_class_idx]*100:.2f}%)")