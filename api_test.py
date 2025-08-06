#!/usr/bin/env python3
"""
Test script for the Rice Leaf Disease Classification API
"""

import requests
import base64
import json
import time
from PIL import Image
import io

# API base URL
BASE_URL = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy!")
            print(f"   Status: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Device: {data['device']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_classes():
    """Test the classes endpoint"""
    print("\n📋 Testing classes endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/classes")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Available classes: {data['classes']}")
            print(f"   Total classes: {data['count']}")
            return data['classes']
        else:
            print(f"❌ Classes endpoint failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Classes endpoint error: {e}")
        return None

def test_file_upload(image_path):
    """Test file upload prediction"""
    print(f"\n📤 Testing file upload with: {image_path}")
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Disease: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Filename: {result['filename']}")
            print(f"   Timestamp: {result['timestamp']}")
            
            print("   All probabilities:")
            for cls, prob in result['probabilities'].items():
                print(f"     {cls}: {prob:.2%}")
            
            return result
        else:
            print(f"❌ File upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ File upload error: {e}")
        return None

def test_base64_upload(image_path):
    """Test base64 encoded image prediction"""
    print(f"\n🔄 Testing base64 upload with: {image_path}")
    try:
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')
        
        # Send base64 data
        payload = {'image': base64_data}
        response = requests.post(f"{BASE_URL}/predict_base64", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Base64 prediction successful!")
            print(f"   Disease: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Timestamp: {result['timestamp']}")
            
            print("   All probabilities:")
            for cls, prob in result['probabilities'].items():
                print(f"     {cls}: {prob:.2%}")
            
            return result
        else:
            print(f"❌ Base64 upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Base64 upload error: {e}")
        return None

def test_batch_predictions():
    """Test multiple predictions from different classes"""
    print("\n🔄 Testing batch predictions...")
    
    # Test images from each class (if available)
    test_images = []
    
    # Try to find one image from each class
    import os
    for class_name in ['Healthy', 'Leafsmut', 'Brownspot', 'Bacterialblight']:
        if os.path.exists(class_name):
            # Get first image from each class
            for file in os.listdir(class_name):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(class_name, file))
                    break
    
    if not test_images:
        print("❌ No test images found in class directories")
        return
    
    print(f"Found {len(test_images)} test images")
    
    results = []
    for i, image_path in enumerate(test_images, 1):
        print(f"\n--- Test {i}/{len(test_images)} ---")
        result = test_file_upload(image_path)
        if result:
            results.append(result)
        time.sleep(1)  # Small delay between requests
    
    # Summary
    print(f"\n📊 Batch Test Summary:")
    print(f"   Total tests: {len(test_images)}")
    print(f"   Successful: {len(results)}")
    print(f"   Failed: {len(test_images) - len(results)}")
    
    if results:
        print("\n   Predictions:")
        for result in results:
            print(f"     {result['filename']}: {result['predicted_class']} ({result['confidence']:.2%})")

def main():
    """Run all tests"""
    print("🚀 Rice Leaf Disease Classification API Test Suite")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health():
        print("❌ API is not healthy. Please make sure the API is running.")
        return
    
    # Test 2: Get classes
    classes = test_classes()
    if not classes:
        print("❌ Could not get classes. API may not be working properly.")
        return
    
    # Test 3: File upload (if test image exists)
    test_image = "Healthy/shape 1.jpg"  # Adjust path as needed
    try:
        test_file_upload(test_image)
    except FileNotFoundError:
        print(f"⚠️  Test image {test_image} not found, skipping file upload test")
    
    # Test 4: Base64 upload (if test image exists)
    try:
        test_base64_upload(test_image)
    except FileNotFoundError:
        print(f"⚠️  Test image {test_image} not found, skipping base64 upload test")
    
    # Test 5: Batch predictions
    test_batch_predictions()
    
    print("\n✅ Test suite completed!")

if __name__ == "__main__":
    main() 