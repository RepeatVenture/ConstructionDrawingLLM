"""
Test Modal Serverless API

Usage:
    # Test with a local image
    python test_modal_api.py path/to/image.png
    
    # Or test with default image
    python test_modal_api.py
"""

import requests
import base64
import json
import sys
from pathlib import Path

# Your Modal API URL (update after deployment)
# Format: https://YOUR_USERNAME--llava-construction-api-analyze.modal.run
API_URL = "https://YOUR_USERNAME--llava-construction-api-analyze.modal.run"

def encode_image(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    health_url = API_URL.replace("-analyze", "-health")
    
    try:
        response = requests.get(health_url, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.ok
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_info():
    """Test info endpoint"""
    print("\nTesting /info endpoint...")
    info_url = API_URL.replace("-analyze", "-info")
    
    try:
        response = requests.get(info_url, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.ok
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_analyze(image_path: str):
    """Test analyze endpoint with an image"""
    print(f"\nTesting /analyze endpoint with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Encode image
    print("Encoding image...")
    image_b64 = encode_image(image_path)
    
    # Prepare request
    request_data = {
        "image": f"data:image/png;base64,{image_b64}",
        "text": """Extract all millwork items from this construction drawing.
        
For each item, identify:
- Room number and name
- Drawing reference
- Category (e.g., cabinet, shelving, door)
- Description
- Finish code
- Quantity and unit

Return as structured JSON with included_items array.""",
        "system_prompt": """You are an expert at reading construction drawings and extracting millwork specifications.
You must respond with valid JSON only, no other text.""",
        "max_tokens": 4000,
        "temperature": 0.7,
        "response_format": "json"
    }
    
    # Send request
    print(f"Sending request to {API_URL}...")
    print(f"Request size: {len(json.dumps(request_data))} bytes")
    
    try:
        response = requests.post(
            API_URL,
            json=request_data,
            timeout=120  # 2 minute timeout
        )
        
        print(f"Status: {response.status_code}")
        
        if response.ok:
            result = response.json()
            
            # Parse content
            if "content" in result:
                content = json.loads(result["content"])
                
                print("\n" + "=" * 70)
                print("EXTRACTION RESULT")
                print("=" * 70)
                
                if "included_items" in content:
                    print(f"\nFound {len(content['included_items'])} items:")
                    for i, item in enumerate(content['included_items'][:5], 1):
                        print(f"\n{i}. {item.get('description', 'N/A')}")
                        print(f"   Room: {item.get('room_number', 'N/A')} - {item.get('room_name', 'N/A')}")
                        print(f"   Category: {item.get('category', 'N/A')}")
                        print(f"   Quantity: {item.get('quantity', 'N/A')} {item.get('unit', 'N/A')}")
                
                if "specifications" in content:
                    print(f"\nSpecifications: {len(content['specifications'])}")
                
                # Show token usage
                if "usage" in result:
                    usage = result["usage"]
                    print(f"\nToken usage:")
                    print(f"  Prompt: {usage.get('prompt_tokens', 0)}")
                    print(f"  Completion: {usage.get('completion_tokens', 0)}")
                    print(f"  Total: {usage.get('total_tokens', 0)}")
                
                # Save full result
                output_file = "modal_api_result.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n✓ Full result saved to {output_file}")
                
                return True
            else:
                print(f"Response: {json.dumps(result, indent=2)}")
                return False
        else:
            print(f"Error response: {response.text}")
            return False
            
    except requests.Timeout:
        print("❌ Request timed out (>120s)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 70)
    print("Modal Serverless API Test")
    print("=" * 70)
    
    # Check if API URL is configured
    if "YOUR_USERNAME" in API_URL:
        print("\n⚠️  Please update API_URL in this script with your Modal endpoint")
        print("After running 'modal deploy modal_serve.py', you'll see URLs like:")
        print("  https://username--llava-construction-api-analyze.modal.run")
        print("\nUpdate the API_URL variable at the top of this script.")
        return
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Find a test image
        test_images = list(Path("training/data/raw").rglob("*.png"))
        if test_images:
            image_path = str(test_images[0])
            print(f"Using default test image: {image_path}")
        else:
            print("❌ No test images found. Please provide an image path:")
            print("   python test_modal_api.py path/to/image.png")
            return
    
    # Run tests
    print(f"\nTesting Modal API at: {API_URL}")
    print("=" * 70)
    
    # Test endpoints
    health_ok = test_health()
    info_ok = test_info()
    analyze_ok = test_analyze(image_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Health:  {'✓ PASS' if health_ok else '❌ FAIL'}")
    print(f"Info:    {'✓ PASS' if info_ok else '❌ FAIL'}")
    print(f"Analyze: {'✓ PASS' if analyze_ok else '❌ FAIL'}")
    
    if all([health_ok, info_ok, analyze_ok]):
        print("\n✓ All tests passed!")
    else:
        print("\n⚠️  Some tests failed")

if __name__ == "__main__":
    main()
