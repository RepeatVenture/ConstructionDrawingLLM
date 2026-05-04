#!/usr/bin/env bash
# Quick smoke test for the server API.
#
# Usage:
#   ./scripts/test_api.sh                          # default localhost:8000
#   ./scripts/test_api.sh http://192.168.1.10:8000  # custom URL
#   ./scripts/test_api.sh http://localhost:8000 YOUR_API_KEY

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
API_KEY="${2:-}"

AUTH_HEADER=""
if [ -n "$API_KEY" ]; then
  AUTH_HEADER="-H \"Authorization: Bearer $API_KEY\""
fi

echo "=== BlueprintBid LLM Server – Smoke Tests ==="
echo "URL: $BASE_URL"
echo ""

# --- Health check ---
echo "1. Health check …"
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""

# --- Create a test image (white 200x200 PNG) ---
TEST_IMG="/tmp/blueprintbid_test_page.png"
python3 -c "
from PIL import Image, ImageDraw, ImageFont
img = Image.new('RGB', (800, 600), 'white')
d = ImageDraw.Draw(img)
d.text((50, 50), 'Room 129 - POC Lab', fill='black')
d.text((50, 100), 'Wall cabinet 17 LF', fill='black')
d.text((50, 150), 'Base cabinet 11 LF', fill='black')
d.text((50, 200), 'Countertop 13-6', fill='black')
d.text((50, 250), 'Finish: PL-1', fill='black')
d.rectangle([400, 50, 750, 300], outline='black', width=2)
d.text((420, 70), 'ELEVATION A', fill='black')
img.save('$TEST_IMG')
print('Test image created: $TEST_IMG')
"
echo ""

# --- Multipart request ---
echo "2. POST /analyze-drawing (multipart) …"
curl -s -X POST "$BASE_URL/analyze-drawing" \
  -F "image=@$TEST_IMG" \
  -F "text=Room 129 POC Lab, Wall cabinet 17 LF, Base cabinet 11 LF, Countertop 13-6, Finish: PL-1" \
  -F "trade=millwork" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  | python3 -m json.tool
echo ""

# --- JSON request ---
echo "3. POST /analyze-drawing-json (JSON/base64) …"
B64=$(python3 -c "import base64; print(base64.b64encode(open('$TEST_IMG','rb').read()).decode())")
curl -s -X POST "$BASE_URL/analyze-drawing-json" \
  -H "Content-Type: application/json" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -d "{
    \"image_base64\": \"$B64\",
    \"text\": \"Wall cabinet 17 LF\",
    \"trade\": \"millwork\"
  }" | python3 -m json.tool
echo ""

# --- Invalid trade ---
echo "4. Invalid trade (expect 400) …"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/analyze-drawing" \
  -F "image=@$TEST_IMG" \
  -F "trade=hvac" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"})
if [ "$HTTP_CODE" = "400" ]; then
  echo "   ✓ Got expected 400"
else
  echo "   ✗ Expected 400, got $HTTP_CODE"
fi
echo ""

# Clean up
rm -f "$TEST_IMG"
echo "=== Done ==="
