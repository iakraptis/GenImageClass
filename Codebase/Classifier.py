import pathlib

import torch
import torch.optim as optim
from torch.nn import BCELoss, Conv2d, Linear, Module, ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


class Classifier(Module):
    # Input images are 1024 * 1024
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = Conv2d(3, 32, 3, 1)
        self.conv2 = Conv2d(32, 32, 3, 1)
        self.fc1 = Linear(492032, 128)
        self.fc2 = Linear(128, 10)
        self.fc3 = Linear(10, 1)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def train_model(
    model: Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    writer: SummaryWriter,
    epochs: int = 10,
):
    criterion = BCELoss()
    optimizer = optim.Adam(model.parameters())
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
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device).float()
                predictions = model(inputs)
                loss = criterion(predictions.view(-1), labels)
                total_loss += loss.item()
            print(f"===> Epoch {epoch + 1}, Total Validation Loss: {total_loss}")
            writer.add_scalar("Loss/validation", total_loss, epoch)

    print("Finished Training")


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


transform = transforms.Compose(
    [
        # Resize to 128x128
        transforms.Resize((128, 128)),
        # Convert to tensor
        transforms.ToTensor(),
        # Normalize if needed (mean, std for each channel)
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


if __name__ == "__main__":
    # Initialize model
    model = Classifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize Train & Test Data
    data_dir = pathlib.Path(__file__).parent.absolute() / "Dataset" / "Images"
    train_dir = data_dir / "Train"
    validation_dir = data_dir / "Validation"
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataset = datasets.ImageFolder(root=validation_dir, transform=transform)
    validation_loader = DataLoader(validation_dataset, batch_size=3, shuffle=False)

    # Initialize Tensorboard
    writer = SummaryWriter()

    # Train Model
    train_model(model, train_loader, validation_loader, writer, epochs=10)

    # Save Model
    # save_model(model, "model.pth")
