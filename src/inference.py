import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image

def predict(image_path, model_path, device):
    # Load the model
    model = EfficientNet.from_pretrained("efficientnet-b0")
    num_features = model._fc.in_features
    model._fc = nn.Linear(num_features, 6)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Preprocess the image
    transform = Compose([
        Resize(512),
        ToTensor(),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

if __name__ == '__main__':
    image_path = "path_to_your_image.jpg"
    model_path = "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prediction = predict(image_path, model_path, device)
    print(f"Predicted class: {prediction}")