#!/usr/bin/env python3
"""
Enhanced training script for better rice leaf disease classification
Includes improved data augmentation, validation, and uncertainty estimation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
import os
from pathlib import Path

# Enhanced data augmentation for better generalization
def get_enhanced_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # Geometric augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        
        # Color augmentations (important for different lighting conditions)
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),  # Simulate poor lighting
        
        # Convert and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # Additional augmentations
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Simulate occlusion
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Improved model architecture with better regularization
class EnhancedRiceClassifier(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(EnhancedRiceClassifier, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze early layers to prevent overfitting
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace classifier with improved architecture
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# Custom loss function with label smoothing
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def train_enhanced_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    train_dir = 'train_data'  # Create this directory structure
    val_dir = 'val_data'      # Create this directory structure
    
    # Get transforms
    train_transform, val_transform = get_enhanced_transforms()
    
    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create model
    model = EnhancedRiceClassifier(num_classes=4).to(device)
    
    # Loss function with label smoothing
    criterion = LabelSmoothingLoss(classes=4, smoothing=0.1)
    
    # Optimizer with different learning rates for different layers
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'backbone.fc' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},      # Lower LR for pretrained layers
        {'params': classifier_params, 'lr': 1e-3}     # Higher LR for new classifier
    ], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    num_epochs = 50
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        val_loss = val_running_loss / len(val_loader)
        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'enhanced_rice_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        print('-' * 60)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('enhanced_training_history.png')
    plt.show()
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    return model

def evaluate_model_uncertainty():
    """Evaluate model with uncertainty estimation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = EnhancedRiceClassifier(num_classes=4).to(device)
    model.load_state_dict(torch.load('enhanced_rice_model.pth'))
    
    # Enable dropout for uncertainty estimation
    def enable_dropout(m):
        if type(m) == nn.Dropout:
            m.train()
    
    model.eval()
    model.apply(enable_dropout)
    
    # Test with multiple forward passes for uncertainty
    def predict_with_uncertainty(image_tensor, n_samples=10):
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                output = torch.softmax(model(image_tensor), dim=1)
                predictions.append(output.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    return predict_with_uncertainty

if __name__ == "__main__":
    print("🚀 Starting Enhanced Rice Leaf Disease Model Training")
    print("=" * 60)
    
    # Create directories if they don't exist
    os.makedirs('train_data', exist_ok=True)
    os.makedirs('val_data', exist_ok=True)
    
    print("📂 Make sure to organize your data as:")
    print("train_data/")
    print("  ├── Healthy/")
    print("  ├── Brownspot/")
    print("  ├── Bacterialblight/")
    print("  └── Leafsmut/")
    print("val_data/")
    print("  ├── Healthy/")
    print("  ├── Brownspot/")
    print("  ├── Bacterialblight/")
    print("  └── Leafsmut/")
    print("")
    
    # Check if data directories exist and have content
    train_path = Path('train_data')
    if not train_path.exists() or not any(train_path.iterdir()):
        print("❌ Please prepare your training data first!")
        print("Copy your dataset images into the train_data and val_data directories.")
    else:
        print("✅ Starting enhanced training...")
        model = train_enhanced_model()
        print("✅ Training completed!")
        print("💡 Your enhanced model is saved as 'enhanced_rice_model.pth'")
        print("📊 Training plots saved as 'enhanced_training_history.png'") 