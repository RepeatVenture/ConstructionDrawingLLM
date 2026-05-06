# Modal Serverless Deployment Guide

Deploy your trained LLaVA model as a serverless API - **pay only when processing requests!**

## 🎯 Benefits of Modal

- ✅ **Pay per use:** ~$0.05 per inference (vs $12/day for EC2)
- ✅ **Auto-scaling:** Scales to zero when idle
- ✅ **No infrastructure:** No servers to manage
- ✅ **Fast deployment:** Deploy in 5 minutes
- ✅ **You already have it:** Same platform you used for training

---

## 📋 Prerequisites

- Modal account (you already have this!)
- Trained model in `trained_model/final/`
- Modal CLI configured (`modal token set --token-id xxx --token-secret xxx`)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Upload Model to Modal

```bash
# Upload your trained adapters to Modal's persistent storage
modal run upload_model_to_modal.py
```

**What this does:**
- Copies `trained_model/final/*` to Modal Volume
- Stores adapter_model.safetensors and adapter_config.json
- Takes ~30 seconds

### Step 2: Deploy API

```bash
# Deploy the serverless API
modal deploy modal_serve.py
```

**What this does:**
- Builds Docker image with dependencies (first time: ~5 minutes)
- Downloads base LLaVA model (cached)
- Deploys web endpoints
- Gives you public URLs

**You'll see output like:**
```
✓ Created web function analyze => https://username--llava-construction-api-analyze.modal.run
✓ Created web function health => https://username--llava-construction-api-health.modal.run
✓ Created web function info => https://username--llava-construction-api-info.modal.run
```

### Step 3: Test It

```bash
# Update test script with your URL
# Edit test_modal_api.py and replace API_URL

# Then test
python test_modal_api.py training/data/raw/Brightview/images/all_pages/sheet_001.png
```

---

## 🔧 Configuration Options

### GPU Choice

Edit `modal_serve.py` line 30:

```python
# Cheaper option (recommended)
GPU_CONFIG = modal.gpu.T4()  # $0.00060/sec (~$0.05 per 8sec inference)

# Faster option
GPU_CONFIG = modal.gpu.A10G()  # $0.00136/sec (~$0.11 per 8sec inference)
```

### Container Idle Timeout

Edit `modal_serve.py` line 52:

```python
container_idle_timeout=300,  # Keep warm for 5 minutes (reduces cold starts)
```

**Options:**
- `60`: More aggressive scaling, lower cost, more cold starts
- `300`: Balanced (recommended)
- `600`: Fewer cold starts, slightly higher cost

---

## 📊 Cost Comparison

**Scenario:** 500 requests/day, 8 seconds per inference

| Configuration | Per Request | Daily | Monthly |
|--------------|-------------|-------|---------|
| **Modal T4** | $0.0048 | $2.40 | $72 |
| **Modal A10G** | $0.0109 | $5.45 | $163 |
| AWS EC2 24/7 | - | $12.62 | $380 |

**Cold start cost:** First request after idle adds ~5-10 seconds (one-time per session)

---

## 🔌 Integration with BlueprintBid

### Next.js Environment Variables

Add to Vercel:

```env
LLAVA_API_URL=https://username--llava-construction-api-analyze.modal.run
LLAVA_API_TIMEOUT=120000
```

### API Client Code

```typescript
// lib/llava-client.ts
export async function analyzeDrawing(
  imageBase64: string,
  prompt: string,
  systemPrompt: string
) {
  const response = await fetch(process.env.LLAVA_API_URL!, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: imageBase64,
      text: prompt,
      system_prompt: systemPrompt,
      max_tokens: 4000,
      temperature: 0.7,
      response_format: 'json'
    }),
    signal: AbortSignal.timeout(120000)
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const result = await response.json();
  return JSON.parse(result.content);
}
```

### Drop-in Replacement

```typescript
// Before (OpenAI)
const openaiResponse = await openai.chat.completions.create({...});
const data = JSON.parse(openaiResponse.choices[0].message.content);

// After (Modal LLaVA)
const modalResponse = await analyzeDrawing(imageB64, prompt, systemPrompt);
const data = modalResponse;  // Already parsed
```

---

## 🧪 Testing

### Test Health Endpoint

```bash
curl https://username--llava-construction-api-health.modal.run
```

Expected:
```json
{
  "status": "healthy",
  "model": "llava-construction-v1",
  "version": "1.0.0"
}
```

### Test Info Endpoint

```bash
curl https://username--llava-construction-api-info.modal.run
```

### Test Analyze (with Python)

```python
import requests
import base64

# Read image
with open('test.png', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Call API
response = requests.post(
    'https://username--llava-construction-api-analyze.modal.run',
    json={
        'image': f'data:image/png;base64,{img_b64}',
        'text': 'Extract millwork items',
        'system_prompt': 'You are an expert...',
        'max_tokens': 4000,
        'temperature': 0.7
    }
)

result = response.json()
data = json.loads(result['content'])
print(data)
```

---

## 📈 Monitoring

### View Logs

```bash
# Stream logs in real-time
modal app logs llava-construction-api

# View recent logs
modal app logs llava-construction-api --lines 100
```

### Check Usage

```bash
# View app stats
modal app list

# Check volume
modal volume list
```

### Monitor in Dashboard

Visit: https://modal.com/apps

- View request counts
- Check latency
- Monitor costs
- See error rates

---

## 🐛 Troubleshooting

### Model Not Loading

```bash
# Check if model was uploaded
modal volume get construction-model-adapters /root/adapters/

# Re-upload if needed
modal run upload_model_to_modal.py
```

### Cold Starts Too Slow

Increase `container_idle_timeout`:
```python
container_idle_timeout=600,  # 10 minutes
```

### Out of Memory

Switch to larger GPU:
```python
GPU_CONFIG = modal.gpu.A10G()  # 24GB instead of 16GB
```

### Timeout Errors

Increase function timeout:
```python
timeout=900,  # 15 minutes (default: 600)
```

---

## 🔒 Security (Optional)

### Add API Key Authentication

Edit `modal_serve.py`:

```python
API_KEY = modal.Secret.from_name("llava-api-key")

@app.function(secrets=[API_KEY])
@modal.web_endpoint(method="POST")
def analyze(request: dict):
    import os
    
    # Check API key
    provided_key = request.get("api_key")
    if provided_key != os.environ["LLAVA_API_KEY"]:
        return {"error": "Invalid API key"}, 401
    
    # ... rest of code
```

Create secret:
```bash
modal secret create llava-api-key LLAVA_API_KEY=your-secret-key
```

---

## 💡 Tips

1. **First request is slow:** Modal downloads base model on first run (cached after)
2. **Keep warm:** Set `container_idle_timeout` to reduce cold starts
3. **Batch processing:** Process multiple images in one request to amortize cold start
4. **Monitor costs:** Check Modal dashboard regularly
5. **Use webhooks:** For async processing, use Modal's webhook support

---

## 🔄 Update Deployment

After making changes:

```bash
# Re-deploy (takes ~30 seconds)
modal deploy modal_serve.py

# No need to re-upload model unless adapters changed
```

---

## 📚 Modal CLI Commands

```bash
# List apps
modal app list

# View logs
modal app logs llava-construction-api

# Stop app (it auto-stops anyway)
modal app stop llava-construction-api

# Delete app
modal app delete llava-construction-api

# List volumes
modal volume list

# Delete volume
modal volume delete construction-model-adapters
```

---

## ✅ Deployment Checklist

- [ ] Modal CLI configured (`modal token set`)
- [ ] Model uploaded (`modal run upload_model_to_modal.py`)
- [ ] API deployed (`modal deploy modal_serve.py`)
- [ ] URLs copied from deployment output
- [ ] Health endpoint tested
- [ ] Test inference successful
- [ ] URLs added to Vercel environment variables
- [ ] BlueprintBid code updated
- [ ] End-to-end test completed

---

## 🆘 Need Help?

1. Check logs: `modal app logs llava-construction-api`
2. View dashboard: https://modal.com/apps
3. Check volume: `modal volume ls construction-model-adapters`
4. Test locally: `modal run modal_serve.py::test`

---

**Ready to deploy? Run these commands:**

```bash
# 1. Upload model
modal run upload_model_to_modal.py

# 2. Deploy API
modal deploy modal_serve.py

# 3. Copy the URLs and test! 🚀
```
