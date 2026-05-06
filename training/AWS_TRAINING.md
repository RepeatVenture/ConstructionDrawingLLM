# AWS Training Guide

Complete guide for training the Construction Drawing LLM on AWS GPU instances.

## Quick Start

```bash
# 1. Configure AWS CLI (if not done)
aws configure

# 2. Create an EC2 key pair (if you don't have one)
aws ec2 create-key-pair \
    --key-name construction-llm \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/construction-llm.pem
chmod 400 ~/.ssh/construction-llm.pem

# 3. Launch training instance
cd training
./aws_train.sh --key construction-llm

# 4. Wait ~15-20 minutes for setup, then SSH in
./aws_manage.sh ssh

# 5. Start training (on the remote instance)
./start_training.sh

# 6. Monitor training
./aws_manage.sh logs

# 7. Download checkpoints when done
./aws_manage.sh download

# 8. Terminate instance
./aws_manage.sh terminate
```

## Prerequisites

### 1. AWS CLI Setup

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure credentials
aws configure
# Enter:
#   AWS Access Key ID
#   AWS Secret Access Key
#   Default region (e.g., us-east-1)
#   Default output format: json
```

### 2. Create EC2 Key Pair

```bash
# Create new key pair
aws ec2 create-key-pair \
    --key-name construction-llm \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/construction-llm.pem

# Set correct permissions
chmod 400 ~/.ssh/construction-llm.pem

# Or use existing key pair
aws ec2 describe-key-pairs
```

### 3. Check AWS Limits

Make sure you have GPU instance limits enabled in your region:

```bash
# Check your service quotas
aws service-quotas list-service-quotas \
    --service-code ec2 \
    --query 'Quotas[?QuotaName==`Running On-Demand G and VT instances`]'

# If limit is 0, request increase at:
# https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas
```

## Instance Types & Costs

| Instance Type | GPU | VRAM | vCPU | RAM | Cost/Hour | Best For |
|---------------|-----|------|------|-----|-----------|----------|
| **g5.xlarge** | A10G | 24GB | 4 | 16GB | $1.006 | **Recommended** |
| g5.2xlarge | A10G | 24GB | 8 | 32GB | $1.212 | Faster training |
| g4dn.xlarge | T4 | 16GB | 4 | 16GB | $0.526 | Budget (7B may fail) |
| p3.2xlarge | V100 | 16GB | 8 | 61GB | $3.060 | Fast but expensive |

**Recommended:** `g5.xlarge` - Best balance of performance and cost (~$4-6 for full training)

## Launch Training Instance

```bash
cd training

# Launch with defaults (g5.xlarge in us-east-1)
./aws_train.sh --key YOUR_KEY_NAME

# Custom instance type
./aws_train.sh --key YOUR_KEY_NAME --instance g5.2xlarge

# Custom region
./aws_train.sh --key YOUR_KEY_NAME --region us-west-2

# Larger storage (default 100GB)
./aws_train.sh --key YOUR_KEY_NAME --volume-size 200
```

### What Happens

1. **Finds Deep Learning AMI** - Latest Ubuntu 22.04 with CUDA/PyTorch
2. **Creates Security Group** - Allows SSH from your IP only
3. **Launches Instance** - With 100GB storage
4. **Installs Dependencies** - Git, Python packages, training requirements
5. **Downloads Model** - Qwen2.5-VL-7B-Instruct (~20 minutes)
6. **Creates Training Script** - Ready to run

## Monitor Setup Progress

```bash
# Check instance status
./aws_manage.sh status

# SSH into instance
./aws_manage.sh ssh

# On the instance, monitor setup
tail -f /var/log/user-data.log

# Wait for the line: "✓ Setup complete!"
```

## Start Training

Once setup completes (~15-20 minutes):

```bash
# SSH into instance
./aws_manage.sh ssh

# Start training in foreground
./start_training.sh

# OR start in background with tmux
tmux new -s train
./start_training.sh
# Press Ctrl+B, then D to detach
```

### Training Parameters

Edit `/home/ubuntu/start_training.sh` on the instance to customize:

```bash
python finetune.py \
    --train-data data/prepared/train.jsonl \
    --val-data data/prepared/val.jsonl \
    --output-dir checkpoints \
    --epochs 3 \              # Training epochs
    --batch-size 2 \          # Increase with more VRAM
    --learning-rate 2e-4 \    # LoRA learning rate
    --save-steps 100          # Save checkpoint every N steps
```

## Monitor Training

### From Your Local Machine

```bash
# Tail training logs
./aws_manage.sh logs

# Check training progress
./aws_manage.sh ssh
# Then on instance:
tail -f ~/ConstructionDrawingLLM/training/training.log
```

### On the Instance

```bash
# Attach to tmux session
tmux attach -t train

# Check GPU usage
nvidia-smi

# Watch GPU usage live
watch -n 1 nvidia-smi

# Check disk space
df -h
```

## Download Checkpoints

```bash
# Download all checkpoints to local machine
./aws_manage.sh download ./my-checkpoints

# Download specific checkpoint
scp -i ~/.ssh/YOUR_KEY.pem -r \
    ubuntu@INSTANCE_IP:~/ConstructionDrawingLLM/training/checkpoints/checkpoint-500 \
    ./checkpoint-500
```

## Cost Management

```bash
# Check current cost
./aws_manage.sh cost

# Terminate immediately when done
./aws_manage.sh terminate

# Or directly
aws ec2 terminate-instances \
    --region us-east-1 \
    --instance-ids i-XXXXX
```

### Expected Costs

**Full training run (3 epochs, ~4-6 hours):**
- g5.xlarge: **$4-6**
- g5.2xlarge: **$5-7**
- p3.2xlarge: **$12-18**

**Don't forget to terminate when done!**

## Training Process

### Timeline

1. **Setup (15-20 min)**: Installing dependencies, downloading model
2. **Training (4-6 hours)**: Fine-tuning with LoRA
   - Epoch 1: ~90-120 minutes
   - Epoch 2: ~90-120 minutes
   - Epoch 3: ~90-120 minutes
3. **Checkpoints saved every 100 steps**

### What Gets Saved

```
checkpoints/
├── checkpoint-100/
│   ├── adapter_model.bin      # LoRA weights
│   ├── adapter_config.json
│   └── training_state.json
├── checkpoint-200/
├── checkpoint-500/
└── final/
    ├── adapter_model.bin
    └── adapter_config.json
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch-size 1

# Use gradient accumulation
--gradient-accumulation-steps 4

# Or use larger instance
./aws_train.sh --key YOUR_KEY --instance g5.2xlarge
```

### Instance Not Responding

```bash
# Check instance state
aws ec2 describe-instances \
    --instance-ids i-XXXXX \
    --query 'Reservations[0].Instances[0].State.Name'

# View console output
aws ec2 get-console-output --instance-id i-XXXXX

# Reboot if needed
aws ec2 reboot-instances --instance-ids i-XXXXX
```

### Training Crashes

```bash
# SSH back in
./aws_manage.sh ssh

# Check logs
tail -100 ~/ConstructionDrawingLLM/training/training.log

# Resume from checkpoint
cd ~/ConstructionDrawingLLM/training
python finetune.py \
    --train-data data/prepared/train.jsonl \
    --resume-from checkpoints/checkpoint-300
```

### Can't Connect via SSH

```bash
# Verify security group allows your current IP
aws ec2 describe-security-groups \
    --group-ids sg-XXXXX

# Update security group if IP changed
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
    --group-id sg-XXXXX \
    --protocol tcp \
    --port 22 \
    --cidr ${MY_IP}/32
```

## After Training

### Test the Model

```bash
# On the instance
cd ~/ConstructionDrawingLLM
python -c "
from transformers import AutoModelForVision2Seq
from peft import PeftModel

# Load base model
base_model = AutoModelForVision2Seq.from_pretrained(
    'Qwen/Qwen2.5-VL-7B-Instruct',
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, 'training/checkpoints/final')
print('✓ Model loaded successfully!')
"
```

### Deploy the Model

See [deployment documentation](../deploy/README.md) for deploying the trained model to production.

## Cleanup

```bash
# Terminate instance
./aws_manage.sh terminate

# Delete security group
aws ec2 delete-security-group \
    --group-id sg-XXXXX

# Clean up local files
rm .aws-training-instance
```

## Support

For issues or questions:
1. Check logs: `./aws_manage.sh logs`
2. SSH in and investigate: `./aws_manage.sh ssh`
3. Review training script: `cat start_training.sh`

## Advanced Usage

### Multiple Training Runs

```bash
# Launch multiple instances with different configs
./aws_train.sh --key key1 --instance g5.xlarge
./aws_train.sh --key key2 --instance g5.2xlarge --region us-west-2

# Each saves to its own .aws-training-instance file
```

### Custom Training Script

Edit `start_training.sh` on the instance:

```bash
# Try different hyperparameters
python finetune.py \
    --train-data data/prepared/train.jsonl \
    --epochs 5 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --warmup-steps 50
```

### Spot Instances (70% savings)

Modify `aws_train.sh` to use spot instances:

```bash
# Add to run-instances command:
--instance-market-options '{
    "MarketType":"spot",
    "SpotOptions":{
        "MaxPrice":"0.50",
        "SpotInstanceType":"one-time"
    }
}'
```

**Note:** Spot instances can be terminated by AWS with 2-minute warning. Good for experimentation, risky for long training runs.
