import requests
import base64
import json

print("Testing Modal API with image (be patient, this takes 5-10 minutes on first run)...")
print("Downloading 14GB LLaVA base model...")

# Read and encode image
with open("training/data/raw/DCI/images/page_003.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Test API with extended timeout
try:
    print("\nSending request (timeout: 15 minutes)...")
    response = requests.post(
        "https://repeatventure--llava-construction-api-fastapi-app.modal.run/analyze",
        json={
            "image": f"data:image/png;base64,{img_b64}",
            "text": "Extract all millwork items from this construction drawing. List each item with room number, description, quantity, and finish code.",
            "system_prompt": "You are an expert at reading construction drawings and extracting millwork specifications. Return valid JSON only.",
            "max_tokens": 2000,
            "temperature": 0.7
        },
        timeout=900  # 15 minute timeout
    )
    
    print(f"\n✓ Got response! Status: {response.status_code}")
    
    if response.ok:
        result = response.json()
        print("\n" + "=" * 70)
        print("SUCCESS! Model is working!")
        print("=" * 70)
        
        # Parse content
        if "content" in result:
            data = json.parse(result["content"])
            items = data.get("included_items", [])
            print(f"\nExtracted {len(items)} items")
            
            if items:
                print("\nFirst 3 items:")
                for i, item in enumerate(items[:3], 1):
                    print(f"\n{i}. {item.get('category', 'N/A')}: {item.get('description', 'N/A')}")
                    print(f"   Room: {item.get('room_number', 'N/A')}")
                    print(f"   Qty: {item.get('quantity', 'N/A')} {item.get('unit', 'N/A')}")
        
        # Show full response
        print("\n" + "=" * 70)
        print("Full API response:")
        print("=" * 70)
        print(json.dumps(result, indent=2)[:1000] + "...")
        
        # Save to file
        with open("modal_test_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print("\n✓ Full result saved to modal_test_result.json")
        
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        
except requests.Timeout:
    print("\n✗ Still timed out after 15 minutes")
    print("Model may still be downloading. Check Modal logs:")
    print("  modal app logs llava-construction-api")
except Exception as e:
    print(f"\n✗ Error: {e}")

