#!/bin/bash
# AWS EC2 Deployment Script for LLaVA Construction Drawing API
# Run this script on your AWS EC2 g4dn.xlarge instance

set -e  # Exit on error

echo "=========================================="
echo "LLaVA Construction Drawing API Deployment"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/home/ubuntu/llava-api"
MODEL_DIR="/home/ubuntu/llava-model"
ADAPTER_DIR="$MODEL_DIR/adapters"
CONDA_ENV="llava"

echo -e "${BLUE}Step 1: System Update and Dependencies${NC}"
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    python3-pip \
    nvidia-utils-525 \
    htop

echo -e "${GREEN}✓ System dependencies installed${NC}"
echo ""

echo -e "${BLUE}Step 2: Install Miniconda (if not installed)${NC}"
if [ ! -d "/home/ubuntu/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p /home/ubuntu/miniconda3
    rm ~/miniconda.sh
    eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"
    conda init
    echo -e "${GREEN}✓ Miniconda installed${NC}"
else
    echo -e "${GREEN}✓ Miniconda already installed${NC}"
fi
echo ""

echo -e "${BLUE}Step 3: Create Python Environment${NC}"
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "Environment $CONDA_ENV exists, removing..."
    conda env remove -n $CONDA_ENV -y
fi
conda create -n $CONDA_ENV python=3.10 -y
conda activate $CONDA_ENV
echo -e "${GREEN}✓ Python environment created${NC}"
echo ""

echo -e "${BLUE}Step 4: Install CUDA PyTorch${NC}"
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
echo -e "${GREEN}✓ PyTorch with CUDA installed${NC}"
echo ""

echo -e "${BLUE}Step 5: Create Project Directory${NC}"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR
echo -e "${GREEN}✓ Project directory created${NC}"
echo ""

echo -e "${BLUE}Step 6: Upload Your Files${NC}"
echo "Please upload the following files to $PROJECT_DIR:"
echo "  1. api_server.py"
echo "  2. requirements-api.txt"
echo ""
echo "Press Enter when files are uploaded..."
read

echo -e "${BLUE}Step 7: Install Python Dependencies${NC}"
pip install -r requirements-api.txt
echo -e "${GREEN}✓ Python dependencies installed${NC}"
echo ""

echo -e "${BLUE}Step 8: Setup Model Directory${NC}"
mkdir -p $ADAPTER_DIR
echo "Please upload your trained model files to $ADAPTER_DIR:"
echo "  1. adapter_model.safetensors"
echo "  2. adapter_config.json"
echo ""
echo "You can use SCP or this command on your local machine:"
echo "  scp -i your-key.pem trained_model/final/* ubuntu@YOUR_EC2_IP:$ADAPTER_DIR/"
echo ""
echo "Press Enter when model files are uploaded..."
read

echo -e "${GREEN}✓ Model directory prepared${NC}"
echo ""

echo -e "${BLUE}Step 9: Update api_server.py Configuration${NC}"
echo "Update MODEL_CONFIG in api_server.py:"
echo "  adapter_path: $ADAPTER_DIR"
echo ""
echo "Press Enter when updated..."
read

echo -e "${BLUE}Step 10: Create Log Directory${NC}"
sudo touch /var/log/llava-api.log
sudo touch /var/log/llava-api-error.log
sudo chown ubuntu:ubuntu /var/log/llava-api.log
sudo chown ubuntu:ubuntu /var/log/llava-api-error.log
echo -e "${GREEN}✓ Log files created${NC}"
echo ""

echo -e "${BLUE}Step 11: Test Server Manually${NC}"
echo "Testing server startup (this will take 2-3 minutes to load model)..."
echo "Press Ctrl+C to stop after you see 'Model loaded and ready'"
python api_server.py
echo ""

echo -e "${BLUE}Step 12: Setup Systemd Service${NC}"
sudo cp llava-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable llava-api
sudo systemctl start llava-api
echo -e "${GREEN}✓ Systemd service configured and started${NC}"
echo ""

echo -e "${BLUE}Step 13: Configure Firewall${NC}"
echo "Ensure port 8000 is open in AWS Security Group:"
echo "  Type: Custom TCP"
echo "  Port: 8000"
echo "  Source: 0.0.0.0/0 (or restrict to your Vercel IPs)"
echo ""

echo -e "${BLUE}Step 14: Verify Deployment${NC}"
sleep 10  # Wait for service to start
echo "Checking service status..."
sudo systemctl status llava-api --no-pager
echo ""
echo "Testing health endpoint..."
curl http://localhost:8000/health
echo ""

echo -e "${GREEN}=========================================="
echo "✓ Deployment Complete!"
echo "==========================================${NC}"
echo ""
echo "Your API is now running at: http://YOUR_EC2_IP:8000"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status llava-api    # Check status"
echo "  sudo systemctl restart llava-api   # Restart service"
echo "  sudo systemctl stop llava-api      # Stop service"
echo "  sudo journalctl -u llava-api -f    # View logs"
echo "  tail -f /var/log/llava-api.log     # View application logs"
echo ""
echo "Next steps:"
echo "  1. Get your EC2 public IP: curl http://169.254.169.254/latest/meta-data/public-ipv4"
echo "  2. Test API: curl http://YOUR_EC2_IP:8000/info"
echo "  3. Update BlueprintBid to use: http://YOUR_EC2_IP:8000/analyze"
