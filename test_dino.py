import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

def extract_dino_features(image_path):
    # Load DINO model
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    dino.eval()
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    
    # Extract features
    with torch.no_grad():
        features = dino(image.unsqueeze(0))
        features = F.normalize(features, dim=-1)
    
    return features.squeeze()

if __name__ == "__main__":
    image_path = input("Enter image path: ")
    features = extract_dino_features(image_path)
    print(f"Feature shape: {features.shape}")
    print(f"Feature sample: {features[:5]}")  # Print first 5 values 