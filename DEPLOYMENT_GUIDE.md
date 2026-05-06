# LLaVA Construction Drawing API - AWS Deployment Guide

Complete guide for deploying your trained LLaVA model on AWS EC2 with GPU.

## 📋 Prerequisites

### AWS EC2 Instance
- **Instance Type:** g4dn.xlarge (Tesla T4, 16GB VRAM, 4 vCPUs, 16GB RAM)
- **AMI:** Deep Learning AMI GPU PyTorch 2.0 (Ubuntu 22.04) or Ubuntu 22.04 LTS
- **Storage:** 100GB EBS volume (at least 50GB for model + dependencies)
- **Security Group:** Port 8000 open for HTTP API access

### Local Requirements
- Trained model adapters in `trained_model/final/`
- SSH key pair for EC2 access
- SCP or similar tool for file transfer

---

## 🚀 Deployment Steps

### 1. Launch EC2 Instance

```bash
# Launch instance (use AWS Console or CLI)
aws ec2 run-instances \
  --image-id ami-xxxxxx \
  --instance-type g4dn.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxx \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100}'
```

**Security Group Rules:**
```
Type: SSH, Port: 22, Source: Your IP
Type: Custom TCP, Port: 8000, Source: 0.0.0.0/0
```

### 2. Connect to Instance

```bash
# Get your EC2 public IP
AWS_IP="YOUR_EC2_PUBLIC_IP"

# SSH into instance
ssh -i your-key.pem ubuntu@$AWS_IP
```

### 3. Upload Deployment Files

From your local machine:

```bash
# Upload API server files
scp -i your-key.pem api_server.py ubuntu@$AWS_IP:~/
scp -i your-key.pem requirements-api.txt ubuntu@$AWS_IP:~/
scp -i your-key.pem llava-api.service ubuntu@$AWS_IP:~/
scp -i your-key.pem deploy_aws.sh ubuntu@$AWS_IP:~/

# Upload trained model
scp -i your-key.pem -r trained_model/final ubuntu@$AWS_IP:~/llava-model/adapters/
```

### 4. Run Deployment Script

On EC2 instance:

```bash
# Make script executable
chmod +x deploy_aws.sh

# Run deployment
./deploy_aws.sh
```

The script will:
1. Install system dependencies
2. Install Miniconda
3. Create Python environment
4. Install PyTorch with CUDA
5. Install API dependencies
6. Setup model directory
7. Configure systemd service
8. Start the API server

**Follow the prompts and press Enter when files are uploaded.**

---

## 🔧 Manual Deployment (Alternative)

If you prefer manual setup:

### Step 1: System Setup

```bash
sudo apt-get update
sudo apt-get install -y build-essential curl git python3-pip nvidia-utils

# Verify GPU
nvidia-smi
```

### Step 2: Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init
source ~/.bashrc
```

### Step 3: Create Environment

```bash
conda create -n llava python=3.10 -y
conda activate llava

# Install PyTorch with CUDA
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Setup Project

```bash
mkdir -p ~/llava-api
cd ~/llava-api

# Upload files here
# Then install dependencies
pip install -r requirements-api.txt
```

### Step 5: Configure Model Path

Edit `api_server.py` and update:

```python
MODEL_CONFIG = {
    "base_model": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "adapter_path": "/home/ubuntu/llava-model/adapters",  # Your path
    "model_name": "llava-construction-v1",
    "version": "1.0.0",
}
```

### Step 6: Test Manually

```bash
# First test - this will download base model (~14GB)
python api_server.py
```

You should see:
```
======================================================================
Loading LLaVA Construction Drawing Model
======================================================================
GPU: Tesla T4 (15.9 GB)
Loading processor: llava-hf/llava-v1.6-vicuna-7b-hf
Loading base model: llava-hf/llava-v1.6-vicuna-7b-hf
Loading LoRA adapters: /home/ubuntu/llava-model/adapters
✓ Model loaded and ready
======================================================================
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Press Ctrl+C to stop.

### Step 7: Setup Systemd Service

```bash
# Copy service file
sudo cp llava-api.service /etc/systemd/system/

# Update paths in service file if needed
sudo nano /etc/systemd/system/llava-api.service

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable llava-api
sudo systemctl start llava-api

# Check status
sudo systemctl status llava-api
```

---

## 🧪 Testing

### Test Locally (on EC2)

```bash
# Upload test script
cd ~/llava-api

# Test health endpoint
curl http://localhost:8000/health

# Test info endpoint
curl http://localhost:8000/info

# Test with image
python test_api.py /path/to/test/image.png
```

### Test Remotely (from your machine)

```bash
# Replace with your EC2 public IP
EC2_IP="YOUR_EC2_PUBLIC_IP"

# Test health
curl http://$EC2_IP:8000/health

# Test info
curl http://$EC2_IP:8000/info

# Test analyze with image (using test script)
# Update API_BASE_URL in test_api.py first
python test_api.py training/data/raw/Brightview/images/all_pages/sheet_001.png
```

---

## 📊 Monitoring

### View Logs

```bash
# Application logs
tail -f /var/log/llava-api.log

# Error logs
tail -f /var/log/llava-api-error.log

# Systemd logs
sudo journalctl -u llava-api -f

# GPU usage
watch -n 1 nvidia-smi
```

### Service Management

```bash
# Check status
sudo systemctl status llava-api

# Restart service
sudo systemctl restart llava-api

# Stop service
sudo systemctl stop llava-api

# Start service
sudo systemctl start llava-api

# View recent logs
sudo journalctl -u llava-api -n 100 --no-pager
```

---

## 🔌 Integration with BlueprintBid

### Update Environment Variables

In your Next.js app on Vercel:

```env
# .env.local or Vercel Environment Variables
LLAVA_API_URL=http://YOUR_EC2_IP:8000
LLAVA_API_TIMEOUT=120000
```

### API Client Code (Next.js)

```typescript
// lib/llava-client.ts
export async function analyzeDrawing(
  imageBase64: string,
  prompt: string,
  systemPrompt: string
) {
  const response = await fetch(`${process.env.LLAVA_API_URL}/analyze`, {
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

### Drop-in Replacement for OpenAI

```typescript
// Before (OpenAI)
const response = await openai.chat.completions.create({
  model: "gpt-4-vision-preview",
  messages: [...]
});

// After (Your LLaVA API)
const response = await analyzeDrawing(imageBase64, prompt, systemPrompt);
```

---

## 🐛 Troubleshooting

### Model Loading Issues

**Problem:** Model fails to load
```bash
# Check GPU
nvidia-smi

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check disk space
df -h

# Check model files exist
ls -lh /home/ubuntu/llava-model/adapters/
```

### Out of Memory

**Problem:** CUDA out of memory
- Reduce `max_tokens` in requests
- Check other processes: `nvidia-smi`
- Restart service: `sudo systemctl restart llava-api`

### Service Won't Start

```bash
# Check logs
sudo journalctl -u llava-api -n 100

# Test manually
cd ~/llava-api
conda activate llava
python api_server.py
```

### Slow Inference

**Problem:** API takes too long
- First request always slower (model warmup)
- Typical: 5-10 seconds per image
- Check GPU utilization: `nvidia-smi`

### Connection Refused

**Problem:** Can't connect from Vercel
- Check Security Group port 8000 is open
- Check service is running: `sudo systemctl status llava-api`
- Check firewall: `sudo ufw status`
- Test locally first: `curl http://localhost:8000/health`

---

## 💰 Cost Optimization

### Instance Costs
- **g4dn.xlarge:** ~$0.526/hour (~$380/month if running 24/7)
- **Stop instance when not in use** to save costs
- **Use spot instances** for ~70% discount (may be interrupted)

### Auto-shutdown Script

```bash
# Add to crontab: stop instance at night
0 22 * * * sudo systemctl stop llava-api
0 8 * * 1-5 sudo systemctl start llava-api
```

---

## 🔒 Security Best Practices

1. **Restrict Security Group:** Only allow your Vercel IPs
2. **Add API Key Authentication:** Modify `api_server.py` to check API key
3. **Use HTTPS:** Setup nginx reverse proxy with Let's Encrypt
4. **Rate Limiting:** Use nginx or FastAPI rate limiting
5. **Monitor Logs:** Setup CloudWatch or similar

---

## 📈 Performance Tuning

### Optimize Inference Speed

```python
# In api_server.py, adjust generation parameters:
output_ids = model.generate(
    **inputs,
    max_new_tokens=request.max_tokens,
    do_sample=False,  # Greedy decode is faster
    num_beams=1,      # No beam search
)
```

### Batch Processing

For multiple images, modify API to accept batches:
```python
@app.post("/analyze_batch")
async def analyze_batch(requests: List[AnalyzeRequest]):
    # Process multiple images in one GPU call
    pass
```

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LLaVA Model Card](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf)
- [AWS EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/g4/)
- [Systemd Service Documentation](https://www.freedesktop.org/software/systemd/man/systemd.service.html)

---

## ✅ Deployment Checklist

- [ ] EC2 instance launched with correct specs
- [ ] Security group configured (port 8000 open)
- [ ] SSH access working
- [ ] Files uploaded (API server, requirements, service file)
- [ ] Model adapters uploaded
- [ ] Python environment created
- [ ] Dependencies installed
- [ ] Model loads successfully
- [ ] Systemd service configured and running
- [ ] Health endpoint responds
- [ ] Test API call successful
- [ ] Integrated with BlueprintBid
- [ ] Monitoring setup

---

## 🆘 Support

If you encounter issues:

1. Check logs: `tail -f /var/log/llava-api.log`
2. Verify GPU: `nvidia-smi`
3. Test manually: `python api_server.py`
4. Check service: `sudo systemctl status llava-api`

Good luck with your deployment! 🚀
