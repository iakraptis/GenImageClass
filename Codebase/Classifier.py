import pathlib

import torch
import torch.optim as optim
from torch.nn import BCELoss, Conv2d, Linear, Module, ReLU, Sigmoid
from torch.utils.data import DataLoader
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

    def train(self, train_loader, epochs=10):
        criterion = BCELoss()
        optimizer = optim.Adam(self.parameters())
        for epoch in range(epochs):
            for i, data in enumerate(train_loader):
                inputs, labels = data
                optimizer.zero_grad()
                predictions = self(inputs)
                labels = labels.float()
                loss = criterion(predictions.view(-1), labels)
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}")
        print("Finished Training")

    def test(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self(images)
                predicted = torch.round(outputs)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Accuracy: {100 * correct / total}%")

    def save(self, path):
        torch.save(self.state_dict(), path)


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
    data_dir = pathlib.Path(__file__).parent.absolute() / "Dataset" / "Images"
    image_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(image_dataset, batch_size=8, shuffle=True)
    model = Classifier()
    model.train(dataloader)
    # model.test(test_loader)
    # model.save("model.pth")
