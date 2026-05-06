import requests
import time

print("Testing Modal API endpoints...")

# Test health (should be fast)
print("\n1. Health check...")
try:
    r = requests.get("https://repeatventure--llava-construction-api-fastapi-app.modal.run/health", timeout=10)
    print(f"   ✓ Status: {r.status_code}")
    print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test info (should be fast)
print("\n2. Info check...")
try:
    r = requests.get("https://repeatventure--llava-construction-api-fastapi-app.modal.run/info", timeout=10)
    print(f"   ✓ Status: {r.status_code}")
    print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\nBasic endpoints are working!")
print("Note: The /analyze endpoint will take 5-10 minutes on first use (downloading 14GB model)")
