#!/bin/bash
# ==========================================================================
#  AWS Training Setup Script
#
#  Launches a GPU EC2 instance and runs fine-tuning
#
#  Prerequisites:
#    - AWS CLI configured (aws configure)
#    - EC2 key pair created
#    - This script creates its own security group
#
#  Usage:
#    ./aws_train.sh --key your-keypair-name
#
#  Instance types (cheapest first):
#    - g4dn.xlarge  : T4 GPU,  16GB VRAM,  $0.526/hr (budget, 3B model only)
#    - g5.xlarge    : A10G,    24GB VRAM,  $1.006/hr (recommended)
#    - g5.2xlarge   : A10G,    24GB VRAM,  $1.212/hr (faster CPU)
#    - p3.2xlarge   : V100,    16GB VRAM,  $3.06/hr  (fast but expensive)
#    - g5.4xlarge   : A10G,    24GB VRAM,  $1.624/hr (multi-GPU)
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Configuration ----
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
INSTANCE_TYPE="g5.xlarge"  # A10G, 24GB VRAM
AMI_ID=""  # Auto-detect Deep Learning AMI
KEY_NAME=""
STACK_NAME="construction-llm-training"
VOLUME_SIZE=100  # GB for model weights + data

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --key)          KEY_NAME="$2";         shift 2 ;;
        --region)       REGION="$2";           shift 2 ;;
        --instance)     INSTANCE_TYPE="$2";    shift 2 ;;
        --volume-size)  VOLUME_SIZE="$2";      shift 2 ;;
        --help|-h)
            echo "Usage: $0 --key YOUR_KEY_PAIR [options]"
            echo ""
            echo "Required:"
            echo "  --key NAME         EC2 key pair name"
            echo ""
            echo "Optional:"
            echo "  --region REGION    AWS region (default: us-east-1)"
            echo "  --instance TYPE    Instance type (default: g5.xlarge)"
            echo "  --volume-size GB   EBS volume size (default: 100)"
            echo ""
            echo "Example:"
            echo "  $0 --key my-keypair --instance g5.xlarge"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---- Validate ----
if [[ -z "$KEY_NAME" ]]; then
    echo "❌ Error: --key is required"
    echo "Usage: $0 --key YOUR_KEY_PAIR"
    exit 1
fi

echo ""
echo "========================================================================"
echo "  AWS Training Instance Setup"
echo "========================================================================"
echo "  Region:        $REGION"
echo "  Instance:      $INSTANCE_TYPE"
echo "  Key Pair:      $KEY_NAME"
echo "  Volume Size:   ${VOLUME_SIZE}GB"
echo "========================================================================"
echo ""

# ---- Find Deep Learning AMI ----
echo "🔍 Finding latest Deep Learning AMI..."
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
              "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)

if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
    echo "❌ Could not find Deep Learning AMI in $REGION"
    exit 1
fi

echo "✓ Found AMI: $AMI_ID"

# ---- Create Security Group ----
echo ""
echo "🔒 Creating security group..."
VPC_ID=$(aws ec2 describe-vpcs \
    --region "$REGION" \
    --filters "Name=is-default,Values=true" \
    --query 'Vpcs[0].VpcId' \
    --output text)

SG_ID=$(aws ec2 create-security-group \
    --region "$REGION" \
    --group-name "$STACK_NAME-sg" \
    --description "Training instance security group" \
    --vpc-id "$VPC_ID" \
    --output text 2>/dev/null || \
    aws ec2 describe-security-groups \
        --region "$REGION" \
        --filters "Name=group-name,Values=$STACK_NAME-sg" \
        --query 'SecurityGroups[0].GroupId' \
        --output text)

# Allow SSH from your IP
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
    --region "$REGION" \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 22 \
    --cidr "${MY_IP}/32" 2>/dev/null || true

echo "✓ Security group: $SG_ID (SSH from ${MY_IP})"

# ---- Create startup script ----
cat > /tmp/user-data.sh << 'EOF'
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
EOF

# ---- Launch instance ----
echo ""
echo "🚀 Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --user-data file:///tmp/user-data.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$STACK_NAME},{Key=Purpose,Value=training}]" \
    --iam-instance-profile Name=ConstructionLLM-TrainingRole 2>/dev/null \
    --query 'Instances[0].InstanceId' \
    --output text) || \
aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --user-data file:///tmp/user-data.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$STACK_NAME},{Key=Purpose,Value=training}]" \
    --query 'Instances[0].InstanceId' \
    --output text

if [[ -z "$INSTANCE_ID" ]]; then
    echo "❌ Failed to launch instance"
    exit 1
fi

echo "✓ Instance launched: $INSTANCE_ID"
echo ""
echo "⏳ Waiting for instance to start..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "========================================================================"
echo "✓ Training instance ready!"
echo "========================================================================"
echo ""
echo "  Instance ID:   $INSTANCE_ID"
echo "  Public IP:     $PUBLIC_IP"
echo "  SSH Command:   ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "========================================================================"
echo "  Setup Progress"
echo "========================================================================"
echo ""
echo "  The instance is now installing dependencies and downloading the model."
echo "  This takes ~15-20 minutes."
echo ""
echo "  To monitor setup progress:"
echo "    ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo "    tail -f /var/log/user-data.log"
echo ""
echo "  When setup is complete, start training:"
echo "    ./start_training.sh"
echo ""
echo "  Or run in background with tmux:"
echo "    tmux new -s train"
echo "    ./start_training.sh"
echo "    # Ctrl+B, then D to detach"
echo "    # tmux attach -t train to reattach"
echo ""
echo "========================================================================"
echo "  Cost Estimate"
echo "========================================================================"
echo ""
case "$INSTANCE_TYPE" in
    g4dn.xlarge)  echo "  ~\$0.53/hour  (~\$2-3 for 3 epochs)" ;;
    g5.xlarge)    echo "  ~\$1.00/hour  (~\$4-6 for 3 epochs)" ;;
    g5.2xlarge)   echo "  ~\$1.21/hour  (~\$5-7 for 3 epochs)" ;;
    p3.2xlarge)   echo "  ~\$3.06/hour  (~\$12-18 for 3 epochs)" ;;
    *)            echo "  Check AWS pricing for $INSTANCE_TYPE" ;;
esac
echo ""
echo "  Remember to terminate when done:"
echo "    aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID"
echo ""
echo "========================================================================"

# Save instance info
cat > "$SCRIPT_DIR/.aws-training-instance" << INFO
INSTANCE_ID=$INSTANCE_ID
REGION=$REGION
PUBLIC_IP=$PUBLIC_IP
KEY_NAME=$KEY_NAME
LAUNCHED=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
INFO

echo ""
echo "✓ Instance info saved to: $SCRIPT_DIR/.aws-training-instance"
