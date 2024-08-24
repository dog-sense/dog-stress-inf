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
    val_split = 0.2  # 20% of the data for validation

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # NOTE: Split Dataset
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # NOTE: SetUp Model Structure
    model = EfficientNet.from_pretrained("efficientnet-b0")

    # NOTE: Get Final Layer
    num_features = model._fc.in_features

    # NOTE: Change Final Layer
    model._fc = nn.Linear(num_features, 6)

    # NOTE: Select Learning Resource
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("USE DEVICE:", device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Start Train")
    # NOTE: Start Train
    model.train()

    best_val_loss = float('inf')
    best_model_wts = None

    for i in range(epoch):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Epoch: {i + 1}, Loss: {train_loss}")

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch: {i + 1}, Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()

        model.train()

        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            print("Best Model Loaded with val loss:", best_val_loss)

        torch.save(model.state_dict(), 'best_model.pth')
