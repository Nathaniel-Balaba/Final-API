import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import argparse

def create_model(num_classes=4):
    """Create the same model architecture as used in training"""
    model = models.resnet50(pretrained=False)
    
    # Replace the final classification layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

def predict_image(model, image_path, device, classes):
    """Predict the class of a single image"""
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return classes[predicted_class], confidence, probabilities[0].cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Predict rice leaf disease from image')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to the trained model')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Please train the model first using 'python train.py'")
        return
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = create_model(num_classes=4)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    
    # Classes
    classes = ['Healthy', 'Leafsmut', 'Brownspot', 'Bacterialblight']
    
    # Make prediction
    predicted_class, confidence, all_probabilities = predict_image(model, args.image, device, classes)
    
    # Print results
    print(f"\nPrediction Results:")
    print(f"Image: {args.image}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    print(f"\nAll Class Probabilities:")
    for i, (class_name, prob) in enumerate(zip(classes, all_probabilities)):
        print(f"  {class_name}: {prob:.2%}")

if __name__ == "__main__":
    main() 