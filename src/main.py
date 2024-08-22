from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

if __name__ == '__main__':
    transform = Compose([
        Resize(512),
        ToTensor(),
    ])

    data_dir = "data"

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    model = EfficientNet.from_pretrained("efficientnet-b0")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for inputs, labels in dataloader:
        print(inputs.shape, labels.shape)
