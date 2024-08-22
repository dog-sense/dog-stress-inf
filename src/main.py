import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor

if __name__ == '__main__':
    transform = Compose([
        Resize(512),
        ToTensor(),
    ])

    data_dir = "data"

    epoch = 10

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # NOTE: SetUp Model Structure
    model = EfficientNet.from_pretrained("efficientnet-b0")

    # NOTE: Get Final Layer
    num_features = model._fc.in_features

    # NOTE: Change Final Layer
    model._fc = nn.Linear(num_features, 6)

    # NOTE: Select Learning Resource
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # NOTE: Start Train
    model.train()

    for i in range(epoch):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch: {i + 1}, Loss: {running_loss / len(dataloader)}")
