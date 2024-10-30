import pathlib
from random import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.nn import (
    BCELoss,
    BCEWithLogitsLoss,
    Linear,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.models import vgg16
from torchvision.utils import make_grid


class FakeImageClassifier(Module):
    def __init__(self):
        super(FakeImageClassifier, self).__init__()

        # Load the VGG16 model and remove its final classification layer
        self.vgg = vgg16(pretrained=True)
        self.vgg.classifier = Sequential(*list(self.vgg.classifier.children())[:-1])
        # # freeze the VGG layers
        # for param in self.vgg.parameters():
        #     param.requires_grad = False

        # Additional layers for binary classification
        self.fc1 = Linear(4096, 128)
        self.fc2 = Linear(128, 1)

        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # Pass input through VGG layers
        x = self.vgg(x)

        # Pass through custom layers for binary classification
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x


def train_model(
    model: Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    writer: SummaryWriter,
    epochs: int = 10,
):
    criterion = BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    writer.add_graph(model, next(iter(train_loader))[0].to(device))
    writer.add_text("Model", str(model))
    for epoch in range(epochs):
        total_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions.view(-1), labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}")
        print(f"===> Epoch {epoch + 1}, Total Train Loss: {total_loss}")
        writer.add_scalar("Loss/train", total_loss, epoch)

        with torch.no_grad():
            total_loss = 0.0
            for i, data in enumerate(validation_loader):
                images, targets = data
                inputs, labels = images.to(device), targets.to(device).float()
                predictions = model(inputs)
                cpu_predictions = predictions.cpu().detach().numpy()
                if random() < 0.1:
                    # add images to tensorboard
                    grid = show_images(images, targets, cpu_predictions)
                    writer.add_figure(f"{epoch},{i}", grid)
                loss = criterion(predictions.view(-1), labels)
                total_loss += loss.item()
            print(f"===> Epoch {epoch + 1}, Total Validation Loss: {total_loss}")
            writer.add_scalar("Loss/validation", total_loss, epoch)

    print("Finished Training")


def show_images(images, labels, predictions):
    # Ensure that batch_size is not larger than the grid capacity (3x3)
    batch_size = min(images.size(0), 9)  # Limit to 9 images for 3x3 grid

    # Create a 3x3 grid
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    fig.tight_layout(pad=3)

    for i in range(batch_size):
        row = i // 3
        col = i % 3
        # Show image and add title with label and prediction
        ax[row, col].imshow(np.transpose(images[i], (1, 2, 0)))
        prer_percentage = predictions[i][0] * 100
        pred_class = "1" if prer_percentage > 50 else "0"
        ax[row, col].set_title(
            f"Label: {labels[i]}, Pred: {pred_class} ({prer_percentage:.2f}%)"
        )
        ax[row, col].axis("off")

    # Hide any unused subplots if batch_size < 9
    for i in range(batch_size, 9):
        fig.delaxes(ax[i // 3, i % 3])

    # plt.show()

    return fig


def test_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")


def save_model(model, path):
    torch.save(model.state_dict(), path)


train_transform = transforms.Compose(
    [
        # Resize to 128x128
        transforms.Resize((512, 512)),
        # Apply random horizontal flip
        transforms.RandomHorizontalFlip(),
        # Apply random rotation
        transforms.RandomRotation(10),
        # Convert to tensor
        transforms.ToTensor(),
        # Normalize if needed (mean, std for each channel)
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
inference_transform = transforms.Compose(
    [
        # Resize to 128x128
        transforms.Resize((512, 512)),
        # Convert to tensor
        transforms.ToTensor(),
        # Normalize if needed (mean, std for each channel)
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


if __name__ == "__main__":
    # Initialize model
    model = FakeImageClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize Train & Test Data
    data_dir = pathlib.Path(__file__).parent.absolute() / "Dataset" / "Images"
    train_dir = data_dir / "Train"
    validation_dir = data_dir / "Validation"
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataset = datasets.ImageFolder(
        root=validation_dir, transform=inference_transform
    )
    validation_loader = DataLoader(validation_dataset, batch_size=9, shuffle=True)

    # Initialize Tensorboard
    writer = SummaryWriter()

    # Train Model
    train_model(model, train_loader, validation_loader, writer, epochs=40)

    # Save Model
    # save_model(model, "model.pth")
