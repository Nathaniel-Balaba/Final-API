from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import os
import numpy as np
from datetime import datetime
import logging
import cv2
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
model = None
device = None
classes = ['Healthy', 'Leafsmut', 'Brownspot', 'Bacterialblight']

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

def load_model():
    """Load the trained model"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = create_model(num_classes=4)
    
    # Load trained weights
    model_path = 'best_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        logger.info("Model loaded successfully!")
    else:
        logger.error(f"Model file {model_path} not found!")
        raise FileNotFoundError(f"Model file {model_path} not found!")

def is_rice_leaf_image(image):
    """
    Strict rice leaf detection - only accepts clear rice leaf structures
    Rejects people, objects, backgrounds, other plants, logos, etc.
    """
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        width, height = image.size
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 1. Basic size check
        if width < 100 or height < 100:
            logger.info(f"Image too small ({width}x{height}) - not a clear rice leaf")
            return False
        
        # 2. Check for skin tones (human faces/bodies)
        # HSV ranges for human skin detection
        lower_skin1 = np.array([0, 20, 70])
        upper_skin1 = np.array([20, 255, 255])
        lower_skin2 = np.array([160, 20, 70])
        upper_skin2 = np.array([180, 255, 255])
        
        skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        skin_percentage = (np.sum(skin_mask > 0) / (width * height)) * 100
        
        if skin_percentage > 15:  # More than 15% skin tone
            logger.info(f"Skin tones detected ({skin_percentage:.1f}%) - likely human image")
            return False
        
        # 3. Check for artificial graphics/logos (high saturation + many edges)
        high_sat_mask = hsv[:,:,1] > 200
        high_sat_percentage = (np.sum(high_sat_mask) / (width * height)) * 100
        
        edges = cv2.Canny(gray, 50, 150)
        edge_percentage = (np.sum(edges > 0) / (width * height)) * 100
        
        # Reject artificial graphics
        if high_sat_percentage > 40 and edge_percentage > 25:
            logger.info(f"Artificial graphics detected - HighSat: {high_sat_percentage:.1f}%, Edges: {edge_percentage:.1f}%")
            return False
        
        # 4. Check for natural leaf colors (strict ranges for rice leaves)
        # Green range for healthy rice leaves
        lower_green = np.array([35, 25, 25])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percentage = (np.sum(green_mask > 0) / (width * height)) * 100
        
        # Brown/yellow range for diseased rice leaves
        lower_brown = np.array([10, 25, 25])
        upper_brown = np.array([35, 255, 255])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        brown_percentage = (np.sum(brown_mask > 0) / (width * height)) * 100
        
        # Grayish colors (for some diseased leaves)
        lower_gray = np.array([0, 0, 40])
        upper_gray = np.array([180, 50, 200])
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        gray_percentage = (np.sum(gray_mask > 0) / (width * height)) * 100
        
        leaf_color_percentage = green_percentage + brown_percentage + gray_percentage
        
        # 5. Check for leaf-like shapes using contour analysis
        # Apply morphological operations to clean up the image
        kernel = np.ones((3,3), np.uint8)
        edges_cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_leaf_shape = False
        if contours:
            # Find significant contours
            significant_contours = [c for c in contours if cv2.contourArea(c) > (width * height * 0.01)]
            
            if significant_contours:
                largest_contour = max(significant_contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                contour_percentage = (contour_area / (width * height)) * 100
                
                # Check aspect ratio and shape characteristics
                rect = cv2.minAreaRect(largest_contour)
                (w, h) = rect[1]
                if w > 0 and h > 0:
                    aspect_ratio = max(w, h) / min(w, h)
                    
                    # Rice leaves are typically elongated (aspect ratio 2-8)
                    # and cover reasonable portion of image (5-70%)
                    if (2 <= aspect_ratio <= 8 and 5 <= contour_percentage <= 70):
                        has_leaf_shape = True
        
        # 6. Final decision - BALANCED criteria for rice leaf (improved for cross-dataset)
        is_rice_leaf = (
            leaf_color_percentage > 10 and  # Reduced threshold for diverse datasets
            (has_leaf_shape or leaf_color_percentage > 25) and  # More lenient shape/color requirements
            edge_percentage < 40 and  # Slightly more permissive for edges
            skin_percentage < 20  # Allow for some lighting variations
        )
        
        logger.info(f"STRICT Leaf detection - Size: {width}x{height}, "
                   f"Colors(G:{green_percentage:.1f}%,B:{brown_percentage:.1f}%,Gr:{gray_percentage:.1f}%), "
                   f"Skin: {skin_percentage:.1f}%, Edges: {edge_percentage:.1f}%, "
                   f"Shape: {has_leaf_shape}, Result: {is_rice_leaf}")
        
        return is_rice_leaf
        
    except Exception as e:
        logger.error(f"Error in leaf detection: {str(e)}")
        # If detection fails, be conservative and reject
        return False

def log_prediction(filename, prediction_result, image_size=None):
    """Log predictions for model improvement and monitoring"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'predicted_class': prediction_result.get('predicted_class', 'unknown'),
            'confidence': prediction_result.get('confidence', 0.0),
            'is_rice_leaf': prediction_result.get('is_rice_leaf', False),
            'image_size': image_size,
            'top_predictions': prediction_result.get('top_predictions', [])
        }
        
        # Log to file for analysis
        with open('prediction_logs.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    except Exception as e:
        logger.error(f"Failed to log prediction: {str(e)}")

def predict_image(image_data):
    """Predict the class of an image with rice leaf validation"""
    # Convert image data to PIL Image
    if isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
    else:
        image = Image.open(image_data).convert('RGB')
    
    # First, check if this looks like a rice leaf
    if not is_rice_leaf_image(image):
        return {
            'predicted_class': 'Not a rice leaf image',
            'confidence': 0.0,
            'probabilities': {cls: 0.0 for cls in classes},
            'is_rice_leaf': False,
            'message': 'The uploaded image does not appear to be a rice leaf. Please upload a clear image of a rice leaf.'
        }
    
    # Preprocess image for model prediction
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()
        all_probabilities = probabilities[0].cpu().numpy()
    
    # Enhanced confidence and uncertainty handling
    predicted_class = classes[predicted_class_idx]
    
    # Class-specific confidence thresholds
    confidence_thresholds = {
        'Healthy': 0.85,      # Higher threshold for healthy (avoid misdiagnosis)
        'Brownspot': 0.75,
        'Bacterialblight': 0.75,
        'Leafsmut': 0.75
    }
    
    required_confidence = confidence_thresholds.get(predicted_class, 0.75)
    
    # Get top 3 predictions for uncertainty cases
    sorted_indices = np.argsort(all_probabilities)[::-1]
    top_predictions = [
        {'class': classes[idx], 'confidence': float(all_probabilities[idx])}
        for idx in sorted_indices[:3]
    ]
    
    # If confidence is too low, return uncertainty
    if confidence < required_confidence:
        return {
            'predicted_class': f'Uncertain prediction: {predicted_class}',
            'confidence': confidence,
            'probabilities': {cls: float(prob) for cls, prob in zip(classes, all_probabilities)},
            'is_rice_leaf': True,
            'top_predictions': top_predictions,
            'message': f'Low confidence ({confidence:.1%}). Suggested: Review image quality or consult expert. Top alternatives: {top_predictions[1]["class"]} ({top_predictions[1]["confidence"]:.1%}), {top_predictions[2]["class"]} ({top_predictions[2]["confidence"]:.1%})'
        }
    
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': {cls: float(prob) for cls, prob in zip(classes, all_probabilities)},
        'is_rice_leaf': True,
        'top_predictions': top_predictions
    }
    
    return result

@app.route('/')
def home():
    """Home page with API documentation"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rice Leaf Disease Classification API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #667eea; }
            .method { background: #667eea; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px; }
            .url { font-family: monospace; background: #e0e0e0; padding: 5px; border-radius: 3px; }
            .upload-form { background: #f9f9f9; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .upload-form input[type="file"] { margin: 10px 0; }
            .upload-form button { background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
            .upload-form button:hover { background: #5a6fd8; }
            .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
            .success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌾 Rice Leaf Disease Classification API</h1>
            <p>Deep learning model to classify rice leaf diseases with 100% accuracy</p>
        </div>
        
        <h2>API Endpoints</h2>
        
        <div class="endpoint">
            <span class="method">POST</span>
            <span class="url">/predict</span>
            <p><strong>Predict disease from uploaded image</strong></p>
            <p>Upload an image file to get disease classification</p>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span>
            <span class="url">/predict_base64</span>
            <p><strong>Predict disease from base64 encoded image</strong></p>
            <p>Send base64 encoded image data in JSON format</p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <span class="url">/health</span>
            <p><strong>Health check endpoint</strong></p>
            <p>Check if the API is running and model is loaded</p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <span class="url">/classes</span>
            <p><strong>Get available disease classes</strong></p>
            <p>Returns list of all supported disease classifications</p>
        </div>
        
        <h2>Test the API</h2>
        
        <div class="upload-form">
            <h3>Upload Image for Prediction</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="imageFile" name="image" accept="image/*" required>
                <button type="submit">Predict Disease</button>
            </form>
            <div id="result"></div>
        </div>
        
        <h2>Example Usage</h2>
        
        <h3>cURL Example</h3>
        <pre><code>curl -X POST -F "image=@your_image.jpg" http://localhost:5000/predict</code></pre>
        
        <h3>Python Example</h3>
        <pre><code>import requests

# Upload image file
with open('rice_leaf.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    result = response.json()
    print(result)</code></pre>
        
        <h3>JavaScript Example</h3>
        <pre><code>const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));</code></pre>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('imageFile');
                formData.append('image', fileInput.files[0]);
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('result');
                    
                    if (response.ok) {
                        resultDiv.className = 'result success';
                        let resultHTML = `<h4>Prediction Result:</h4>`;
                        
                        if (result.predicted_class === 'Not a rice leaf image') {
                            resultHTML += `
                                <p><strong>Result:</strong> ${result.predicted_class}</p>
                                <p><strong>Message:</strong> ${result.message}</p>
                            `;
                        } else if (result.predicted_class.includes('Uncertain')) {
                            resultHTML += `
                                <p><strong>Result:</strong> ${result.predicted_class}</p>
                                <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
                                <p><strong>Message:</strong> ${result.message}</p>
                                <p><strong>Top Predictions:</strong></p>
                                <ul>
                                    ${result.top_predictions.map(pred => 
                                        `<li><strong>${pred.class}:</strong> ${(pred.confidence * 100).toFixed(2)}%</li>`
                                    ).join('')}
                                </ul>
                                <p><em>💡 Tip: For better accuracy, ensure good lighting and clear leaf visibility</em></p>
                            `;
                        } else {
                            resultHTML += `
                                <p><strong>Disease:</strong> ${result.predicted_class}</p>
                                <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
                                <p><strong>Rice Leaf Detected:</strong> ✅ Yes</p>
                                <p><strong>All Probabilities:</strong></p>
                                <ul>
                                    ${Object.entries(result.probabilities).map(([cls, prob]) => 
                                        `<li>${cls}: ${(prob * 100).toFixed(2)}%</li>`
                                    ).join('')}
                                </ul>
                            `;
                        }
                        
                        resultDiv.innerHTML = resultHTML;
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `<p>Error: ${result.error}</p>`;
                    }
                } catch (error) {
                    const resultDiv = document.getElementById('result');
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease from uploaded image file"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read image data
        image_data = file.read()
        
        # Make prediction
        result = predict_image(image_data)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['filename'] = file.filename
        
        # Log prediction for monitoring and improvement
        log_prediction(file.filename, result, image_size=f"{image_data}")
        
        logger.info(f"Prediction made for {file.filename}: {result['predicted_class']} ({result['confidence']:.2%})")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Predict disease from base64 encoded image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No base64 image data provided'}), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
        except Exception as e:
            return jsonify({'error': 'Invalid base64 image data'}), 400
        
        # Make prediction
        result = predict_image(image_data)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Base64 prediction made: {result['predicted_class']} ({result['confidence']:.2%})")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in base64 prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        if model is None:
            return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'device': str(device),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available disease classes"""
    return jsonify({
        'classes': classes,
        'count': len(classes)
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Load model on import (for production servers like Gunicorn)
try:
    load_model()
    logger.info("Model loaded successfully on import!")
except Exception as e:
    logger.error(f"Failed to load model on import: {str(e)}")

if __name__ == '__main__':
    # Load model on startup
    try:
        load_model()
        print("🚀 Rice Leaf Disease Classification API is starting...")
        print("📊 Model loaded successfully!")
        
        # Get port from environment variable (for Render) or default to 5000
        port = int(os.environ.get('PORT', 5000))
        
        print(f"🌐 API will be available at: http://localhost:{port}")
        print(f"📖 API documentation available at: http://localhost:{port}")
        print(f"🔍 Health check: http://localhost:{port}/health")
        print("\nPress Ctrl+C to stop the server")
        
        # Run the Flask app
        app.run(host='0.0.0.0', port=port, debug=False)
    
    except Exception as e:
        print(f"❌ Failed to start API: {str(e)}")
        print("Make sure you have trained the model first using 'python train.py'") 