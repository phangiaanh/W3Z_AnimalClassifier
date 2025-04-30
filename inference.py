import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import os

class AnimalClassifier(nn.Module):
    def __init__(self, input_dim=384):
        super(AnimalClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )
    
    def forward(self, x):
        return self.classifier(x)

def load_model(model_path, device):
    model = AnimalClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(image_path, model, device):
    # Load DINO model
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    dino.eval()
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    
    # Extract DINO features
    with torch.no_grad():
        features = dino(image_tensor.unsqueeze(0).to(device))
        features = F.normalize(features, dim=-1)
        
        # Get model prediction
        outputs = model(features)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Map class index to label (corrected alphabetical order with Hippo)
    class_names = ['Bovidae', 'Canidae', 'Equidae', 'Felidae', 'Hippopotamidae']
    predicted_label = class_names[predicted_class]
    
    return predicted_label, confidence

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_path = input("Enter path to model weights (.pth file): ")
    model = load_model(model_path, device)
    
    # Get image path
    image_path = input("Enter path to image for prediction: ")
    
    # Make prediction
    predicted_label, confidence = predict_image(image_path, model, device)
    
    print(f"\nPrediction Results:")
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main() 