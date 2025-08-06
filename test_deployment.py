#!/usr/bin/env python3
"""
Simple test script to verify the deployment works
"""
import requests
import sys
import time

def test_health_endpoint(base_url):
    """Test the health endpoint"""
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed with error: {e}")
        return False

def test_classes_endpoint(base_url):
    """Test the classes endpoint"""
    try:
        response = requests.get(f"{base_url}/classes", timeout=30)
        if response.status_code == 200:
            print("✅ Classes endpoint works")
            print(f"Available classes: {response.json()['classes']}")
            return True
        else:
            print(f"❌ Classes endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Classes endpoint failed with error: {e}")
        return False

def main():
    if len(sys.argv) > 1:
        base_url = sys.argv[1].rstrip('/')
    else:
        base_url = "http://localhost:5000"
    
    print(f"Testing deployment at: {base_url}")
    print("=" * 50)
    
    # Wait a moment for the server to be ready
    print("Waiting 5 seconds for server to be ready...")
    time.sleep(5)
    
    tests_passed = 0
    total_tests = 2
    
    # Test health endpoint
    if test_health_endpoint(base_url):
        tests_passed += 1
    
    # Test classes endpoint
    if test_classes_endpoint(base_url):
        tests_passed += 1
    
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Deployment is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 