#!/bin/bash
set -x
exec > >(tee /var/log/user-data.log) 2>&1

echo "Starting training instance setup..."

# Update system
apt-get update
apt-get install -y git htop tmux

# Clone repository
cd /home/ubuntu
if [ ! -d "ConstructionDrawingLLM" ]; then
    sudo -u ubuntu git clone https://github.com/RepeatVenture/ConstructionDrawingLLM.git
    cd ConstructionDrawingLLM
else
    cd ConstructionDrawingLLM
    sudo -u ubuntu git pull
fi

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r training/requirements-training.txt

# Download model weights (takes ~20 min)
python3 << 'PYTHON'
from transformers import AutoProcessor, AutoModelForVision2Seq
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
print(f"Downloading {model_id}...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True)
print("✓ Model downloaded to cache")
PYTHON

# Create training script
cat > /home/ubuntu/start_training.sh << 'TRAIN'
#!/bin/bash
cd /home/ubuntu/ConstructionDrawingLLM/training

python finetune.py \
    --train-data data/prepared/train.jsonl \
    --val-data data/prepared/val.jsonl \
    --output-dir checkpoints \
    --epochs 3 \
    --batch-size 2 \
    --learning-rate 2e-4 \
    --save-steps 100 \
    2>&1 | tee training.log
TRAIN

chmod +x /home/ubuntu/start_training.sh
chown ubuntu:ubuntu /home/ubuntu/start_training.sh

echo ""
echo "========================================="
echo "✓ Setup complete!"
echo ""
echo "To start training:"
echo "  1. SSH into this instance"
echo "  2. Run: ./start_training.sh"
echo ""
echo "Or run in background:"
echo "  tmux new -s train"
echo "  ./start_training.sh"
echo "  # Press Ctrl+B, then D to detach"
echo "========================================="
