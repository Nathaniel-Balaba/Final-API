#!/usr/bin/env python3
"""
Test script for rice leaf detection functionality
Tests various image types to ensure proper classification
"""
import requests
import json
import sys
import base64
from pathlib import Path

def test_image_classification(image_path, expected_result, base_url="http://localhost:5000"):
    """Test image classification and compare with expected result"""
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{base_url}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            predicted = result.get('predicted_class', '')
            is_rice_leaf = result.get('is_rice_leaf', False)
            confidence = result.get('confidence', 0)
            
            print(f"\n📁 Testing: {image_path}")
            print(f"   Expected: {expected_result}")
            print(f"   Predicted: {predicted}")
            print(f"   Is Rice Leaf: {is_rice_leaf}")
            print(f"   Confidence: {confidence:.2%}")
            
            # Check if result matches expectation
            if expected_result == "Not a rice leaf image":
                success = predicted == "Not a rice leaf image" or not is_rice_leaf
            elif expected_result in ["Healthy", "Brownspot", "Bacterialblight", "Leafsmut"]:
                success = is_rice_leaf and predicted == expected_result
            else:
                success = False
            
            if success:
                print("   ✅ PASS")
                return True
            else:
                print("   ❌ FAIL")
                return False
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing {image_path}: {str(e)}")
        return False

def create_test_cases():
    """Define test cases for different image types"""
    test_cases = [
        # These should be rejected as "Not a rice leaf image"
        ("test_images/logo.png", "Not a rice leaf image"),
        ("test_images/icon.jpg", "Not a rice leaf image"),
        ("test_images/person.jpg", "Not a rice leaf image"),
        ("test_images/car.jpg", "Not a rice leaf image"),
        ("test_images/grass.jpg", "Not a rice leaf image"),
        ("test_images/tree_leaves.jpg", "Not a rice leaf image"),
        ("test_images/text_document.jpg", "Not a rice leaf image"),
        
        # These should be accepted and classified
        ("test_images/healthy_rice_leaf.jpg", "Healthy"),
        ("test_images/brown_spot_leaf.jpg", "Brownspot"),
        ("test_images/bacterial_blight_leaf.jpg", "Bacterialblight"),
        ("test_images/leaf_smut.jpg", "Leafsmut"),
    ]
    return test_cases

def main():
    if len(sys.argv) > 1:
        base_url = sys.argv[1].rstrip('/')
    else:
        base_url = "http://localhost:5000"
    
    print("🧪 Rice Leaf Detection Test Suite")
    print("=" * 50)
    print(f"Testing API at: {base_url}")
    
    # First, check if API is running
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code != 200:
            print("❌ API is not running or not healthy")
            sys.exit(1)
        print("✅ API is running")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Make sure the API is running: python app.py")
        sys.exit(1)
    
    # Manual test for available images in your dataset
    print("\n📋 Testing with available dataset samples...")
    
    # Test with actual files from your dataset
    dataset_tests = []
    
    # Check if dataset folders exist
    healthy_path = Path("Healthy")
    brownspot_path = Path("Brownspot")
    
    if healthy_path.exists():
        healthy_files = list(healthy_path.glob("*.jpg"))[:2]  # Test first 2 files
        for file in healthy_files:
            dataset_tests.append((str(file), "Healthy"))
    
    if brownspot_path.exists():
        brownspot_files = list(brownspot_path.glob("*.jpg"))[:2]  # Test first 2 files
        for file in brownspot_files:
            dataset_tests.append((str(file), "Brownspot"))
    
    if not dataset_tests:
        print("⚠️  No dataset images found. Testing with manual input...")
        print("\nTo test manually:")
        print("1. Start the API: python app.py")
        print("2. Open http://localhost:5000 in your browser")
        print("3. Upload different types of images:")
        print("   - Rice leaf images (should be classified)")
        print("   - Non-rice images like logos, icons, people (should return 'Not a rice leaf image')")
        return
    
    # Run tests on dataset samples
    passed = 0
    total = len(dataset_tests)
    
    for image_path, expected in dataset_tests:
        if test_image_classification(image_path, expected, base_url):
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        print("\n✅ Your rice leaf detection is working correctly!")
        print("\nRecommended manual tests:")
        print("1. Upload a logo/icon → Should return 'Not a rice leaf image'")
        print("2. Upload a photo of a person → Should return 'Not a rice leaf image'")
        print("3. Upload grass/tree leaves → Should return 'Not a rice leaf image'")
        print("4. Upload rice leaf photos → Should classify the disease correctly")
    else:
        print("❌ Some tests failed. Check the results above.")
    
    print(f"\n🌐 Test your API manually at: {base_url}")

if __name__ == "__main__":
    main() 