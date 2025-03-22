import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from PIL import Image
import os

class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None, cache_dir=None, chunk_size=1000):
        self.dataset = ImageFolder(root_dir)
        self.transform = transform
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        
        # Load DINO model
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.dino.eval()
        
        # Create cache directory if specified
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.process_and_cache_features()
            
    def process_and_cache_features(self):
        """Process images in chunks and cache DINO features"""
        total_samples = len(self.dataset)
        
        for start_idx in range(0, total_samples, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_samples)
            chunk_name = f"features_{start_idx}_{end_idx}.pt"
            cache_path = os.path.join(self.cache_dir, chunk_name)
            
            # Skip if chunk already processed
            if os.path.exists(cache_path):
                continue
                
            features_list = []
            labels_list = []
            
            print(f"Processing samples {start_idx} to {end_idx}...")
            for idx in range(start_idx, end_idx):
                img_path, label = self.dataset.samples[idx]
                image = Image.open(img_path).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                
                # Extract DINO features
                with torch.no_grad():
                    features = self.dino(image.unsqueeze(0))
                    features = F.normalize(features, dim=-1)
                
                features_list.append(features.squeeze())
                labels_list.append(label)
            
            # Save chunk to disk
            torch.save({
                'features': torch.stack(features_list),
                'labels': torch.tensor(labels_list)
            }, cache_path)
            
            # Clear memory
            del features_list, labels_list
            torch.cuda.empty_cache()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.cache_dir:
            # Calculate which chunk contains this index
            chunk_start = (idx // self.chunk_size) * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, len(self.dataset))
            chunk_name = f"features_{chunk_start}_{chunk_end}.pt"
            cache_path = os.path.join(self.cache_dir, chunk_name)
            
            # Load cached features
            chunk_data = torch.load(cache_path)
            chunk_idx = idx - chunk_start
            return chunk_data['features'][chunk_idx], chunk_data['labels'][chunk_idx]
        else:
            # Original processing if no cache
            img_path, label = self.dataset.samples[idx]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            with torch.no_grad():
                features = self.dino(image.unsqueeze(0))
                features = F.normalize(features, dim=-1)
            
            return features.squeeze(), label

class AnimalClassifier(nn.Module):
    def __init__(self, input_dim=384):  # ViT-S/16 DINO features dimension
        super(AnimalClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5)  # 5 animal categories
        )
    
    def forward(self, x):
        return self.classifier(x)

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50):
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_animal_classifier.pth')

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transform for input images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Dataset and DataLoader
    train_dir = input("Enter the path to training directory: ")
    val_dir = input("Enter the path to validation directory: ")
    
    # Add cache directories
    train_cache_dir = "train_features_cache"
    val_cache_dir = "val_features_cache"
    
    train_dataset = AnimalDataset(train_dir, transform=transform, 
                                cache_dir=train_cache_dir, chunk_size=1000)
    val_dataset = AnimalDataset(val_dir, transform=transform, 
                              cache_dir=val_cache_dir, chunk_size=1000)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model, loss function, and optimizer
    model = AnimalClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, device)
    print("Training completed! Best model saved as 'best_animal_classifier.pth'")

if __name__ == "__main__":
    main()
