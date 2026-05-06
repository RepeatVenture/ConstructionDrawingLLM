#!/usr/bin/env python3
"""
Test script for LLaVA Construction Drawing API
"""
import requests
import json
import base64
from pathlib import Path
import sys

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change to your EC2 IP for remote testing

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 data URI."""
    with open(image_path, "rb") as f:
        image_data = f.read()
    b64_data = base64.b64encode(image_data).decode('utf-8')
    
    # Detect image format
    ext = Path(image_path).suffix.lower()
    mime_type = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
    }.get(ext, 'image/png')
    
    return f"data:{mime_type};base64,{b64_data}"


def test_health():
    """Test health endpoint."""
    print("=" * 70)
    print("Testing Health Endpoint")
    print("=" * 70)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_info():
    """Test info endpoint."""
    print("\n" + "=" * 70)
    print("Testing Info Endpoint")
    print("=" * 70)
    
    try:
        response = requests.get(f"{API_BASE_URL}/info", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_analyze(image_path: str):
    """Test analyze endpoint with an image."""
    print("\n" + "=" * 70)
    print("Testing Analyze Endpoint")
    print("=" * 70)
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        # Encode image
        print(f"Encoding image: {image_path}")
        image_b64 = encode_image_to_base64(image_path)
        print(f"Image size: {len(image_b64)} characters")
        
        # Prepare request
        request_data = {
            "image": image_b64,
            "text": """Analyze this construction drawing. Extract all millwork scope items and specifications.

Look for:
1. Visual elements (shapes, symbols, layouts, dimensions)
2. Text annotations and callouts
3. Schedules and tables
4. Detail drawings and sections""",
            "system_prompt": """You are a professional millwork estimator extracting scope from construction drawings.

Extract millwork components in this JSON format:

{
  "included_items": [
    {
      "room_number": "string",
      "room_name": "string",
      "drawing_refs": ["string"],
      "category": "Cabinetry",
      "description": "detailed description",
      "finish_code": "PL-1",
      "count": 5,
      "quantity": 13.5,
      "unit": "l/f",
      "confidence": 0.95
    }
  ],
  "specifications": [
    {
      "key": "Cabinet Finish PL-1",
      "value": "Plastic laminate, white",
      "confidence": 0.9
    }
  ]
}""",
            "max_tokens": 2000,
            "temperature": 0.7,
            "response_format": "json"
        }
        
        # Send request
        print("\nSending request to API...")
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json=request_data,
            timeout=120  # 2 minute timeout
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nModel:", result['model'])
            print("Tokens:", result['usage'])
            
            # Parse content
            content = json.loads(result['content'])
            print("\n" + "=" * 70)
            print("RESULTS")
            print("=" * 70)
            print(json.dumps(content, indent=2))
            
            print(f"\n✓ Found {len(content.get('included_items', []))} items")
            print(f"✓ Found {len(content.get('specifications', []))} specifications")
            return True
        else:
            print(f"❌ Request failed")
            print(response.text)
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (model may still be loading)")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("LLaVA Construction Drawing API Test Suite")
    print("=" * 70)
    print(f"API URL: {API_BASE_URL}")
    print()
    
    # Test health
    if not test_health():
        print("\n❌ Health check failed. Is the server running?")
        sys.exit(1)
    
    # Test info
    if not test_info():
        print("\n❌ Info endpoint failed")
        sys.exit(1)
    
    # Test analyze if image provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not test_analyze(image_path):
            print("\n❌ Analyze endpoint failed")
            sys.exit(1)
    else:
        print("\n⚠️  Skipping analyze test (no image provided)")
        print("Usage: python test_api.py [path/to/drawing.png]")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
