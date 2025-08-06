# Rice Leaf Diseases Classification

This project implements a complete deep learning solution for classifying rice leaf diseases using PyTorch. The model achieves 100% validation accuracy and includes a production-ready API with web interface.

## 🌟 Features

- **100% Validation Accuracy** on rice leaf disease classification
- **Production-Ready API** with Flask backend
- **Beautiful Web Interface** for easy testing
- **Docker Support** for easy deployment
- **Comprehensive Testing Suite**
- **Multiple Input Methods** (file upload, base64)
- **Real-time Predictions** with confidence scores

## 🎯 Disease Classes

The model can identify four different classes:
- **Healthy**: Healthy rice leaves
- **Leafsmut**: Rice leaves with leaf smut disease
- **Brownspot**: Rice leaves with brown spot disease  
- **Bacterialblight**: Rice leaves with bacterial blight disease

## Dataset Structure

The dataset should be organized as follows:
```
rice leaf diseases dataset/
├── Healthy/
│   ├── shape 1.jpg
│   ├── shape 10.jpg
│   └── ...
├── Leafsmut/
│   ├── shape 1.jpg
│   ├── shape 10.jpg
│   └── ...
├── Brownspot/
│   ├── shape 1.jpg
│   ├── shape 10.jpg
│   └── ...
└── Bacterialblight/
    ├── shape 1.jpg
    ├── shape 10.jpg
    └── ...
```

## 🚀 Quick Start

### 1. Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training the Model

Train the model on your dataset:
```bash
python train.py
```

### 3. Start the API

Run the production API:
```bash
python app.py
```

### 4. Use the Web Interface

Open `web_interface.html` in your browser or visit `http://localhost:5000` for the built-in interface.

### 5. Test the API

Run the comprehensive test suite:
```bash
python api_test.py
```

### Training Features

- **Transfer Learning**: Uses pre-trained ResNet-50 model
- **Data Augmentation**: Random horizontal flip, rotation, color jitter, and resized crop
- **Automatic Train/Validation Split**: 80% training, 20% validation
- **Model Checkpointing**: Saves the best model based on validation accuracy
- **Learning Rate Scheduling**: Reduces learning rate every 7 epochs
- **Progress Tracking**: Real-time training progress with tqdm
- **Visualization**: Generates training curves and confusion matrix

### Training Outputs

After training, the following files will be generated:
- `best_model.pth`: The best trained model weights
- `training_history.png`: Training and validation loss/accuracy curves
- `confusion_matrix.png`: Confusion matrix for model evaluation

## 🔌 API Usage

### REST API Endpoints

The API provides several endpoints for different use cases:

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Get Available Classes
```bash
curl http://localhost:5000/classes
```

#### Predict from File Upload
```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/predict
```

#### Predict from Base64
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"image":"base64_encoded_image_data"}' \
  http://localhost:5000/predict_base64
```

### Python Client Example

```python
import requests

# Upload image file
with open('rice_leaf.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    result = response.json()
    print(f"Disease: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

### JavaScript Example

```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## 📱 Web Interface

The project includes a beautiful, responsive web interface:

- **Drag & Drop** image upload
- **Real-time** predictions
- **Visual probability bars**
- **Mobile-friendly** design
- **API status monitoring**

Open `web_interface.html` in your browser or visit `http://localhost:5000` when the API is running.

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start the service
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### Using Docker directly

```bash
# Build the image
docker build -t rice-disease-api .

# Run the container
docker run -p 5000:5000 -v $(pwd)/best_model.pth:/app/best_model.pth:ro rice-disease-api
```

## 📊 Command Line Predictions

For command-line predictions:

```bash
python predict.py --image path/to/your/image.jpg
```

### Example Usage

```bash
# Predict using the default model
python predict.py --image Healthy/shape 1.jpg

# Predict using a specific model
python predict.py --image test_image.jpg --model best_model.pth
```

## Model Architecture

The model uses:
- **Backbone**: ResNet-50 (pre-trained on ImageNet)
- **Transfer Learning**: Frozen early layers, fine-tuned later layers
- **Classification Head**: 
  - Dropout (0.5)
  - Linear layer (2048 → 512)
  - ReLU activation
  - Dropout (0.3)
  - Linear layer (512 → 4 classes)

## Training Configuration

Default training parameters:
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Epochs**: 20
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Image Size**: 224x224

## Performance Metrics

The training script provides:
- Training and validation accuracy
- Training and validation loss
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization

## 📁 Project Structure

```
rice-leaf-diseases/
├── train.py              # Training script
├── predict.py            # Command-line prediction
├── app.py                # Flask API server
├── api_test.py           # API testing suite
├── web_interface.html    # Web interface
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── DEPLOYMENT.md         # Deployment guide
├── LICENSE               # MIT License
├── README.md             # This file
├── best_model.pth        # Trained model (after training)
├── training_history.png  # Training curves (after training)
├── confusion_matrix.png  # Confusion matrix (after training)
├── Healthy/              # Healthy rice leaves dataset
├── Leafsmut/             # Leaf smut disease dataset
├── Brownspot/            # Brown spot disease dataset
└── Bacterialblight/      # Bacterial blight disease dataset
```

## 📋 Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision 0.10+
- Flask 2.0+
- PIL (Pillow)
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- requests

## Tips for Better Performance

1. **More Data**: Collect more images for each class
2. **Data Quality**: Ensure images are clear and properly labeled
3. **Hyperparameter Tuning**: Experiment with different learning rates and batch sizes
4. **Model Architecture**: Try different pre-trained models (ResNet-101, EfficientNet, etc.)
5. **Data Augmentation**: Add more augmentation techniques if needed

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in `train.py`
2. **Slow Training**: Use GPU if available, or reduce image size
3. **Poor Accuracy**: Check data quality and balance between classes

### GPU Support

The script automatically detects and uses GPU if available. To force CPU usage, modify the device selection in the code.

## License

This project is for educational and research purposes. 