import requests
import base64
import json

# Read and encode image
with open("training/data/raw/DCI/images/page_003.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Test API
print("Testing LLaVA API...")
response = requests.post(
    "https://repeatventure--llava-construction-api-fastapi-app.modal.run/analyze",
    json={
        "image": f"data:image/png;base64,{img_b64}",
        "text": "Extract all millwork items from this construction drawing.",
        "system_prompt": "You are an expert at reading construction drawings.",
        "max_tokens": 2000,
        "temperature": 0.7
    },
    timeout=120
)

print(f"Status: {response.status_code}")
if response.ok:
    result = response.json()
    print("\nResult:")
    print(json.dumps(result, indent=2))
else:
    print(f"Error: {response.text}")
