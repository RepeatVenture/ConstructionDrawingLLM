#!/bin/bash
# ==========================================================================
#  AWS Training Instance Management
#
#  Check status, SSH, download checkpoints, or terminate training instances
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFO_FILE="$SCRIPT_DIR/.aws-training-instance"

# Load instance info
if [[ ! -f "$INFO_FILE" ]]; then
    echo "❌ No training instance found"
    echo "   Launch one with: ./aws_train.sh"
    exit 1
fi

source "$INFO_FILE"

# ---- Commands ----
case "${1:-status}" in
    status)
        echo "Training Instance Status"
        echo "========================================================================"
        aws ec2 describe-instances \
            --region "$REGION" \
            --instance-ids "$INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].{State:State.Name,IP:PublicIpAddress,Type:InstanceType,Launched:LaunchTime}' \
            --output table
        ;;
    
    ssh)
        echo "Connecting to $PUBLIC_IP..."
        ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}
        ;;
    
    logs)
        echo "Fetching training logs..."
        ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} "tail -f ~/ConstructionDrawingLLM/training/training.log"
        ;;
    
    setup-logs)
        echo "Fetching setup logs..."
        ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} "tail -f /var/log/user-data.log"
        ;;
    
    download)
        CHECKPOINT_DIR="${2:-./checkpoints}"
        echo "Downloading checkpoints to: $CHECKPOINT_DIR"
        mkdir -p "$CHECKPOINT_DIR"
        scp -i ~/.ssh/${KEY_NAME}.pem -r \
            ubuntu@${PUBLIC_IP}:~/ConstructionDrawingLLM/training/checkpoints/* \
            "$CHECKPOINT_DIR/"
        echo "✓ Checkpoints downloaded"
        ;;
    
    terminate)
        echo "Terminating instance $INSTANCE_ID..."
        read -p "Are you sure? (yes/no): " confirm
        if [[ "$confirm" == "yes" ]]; then
            aws ec2 terminate-instances \
                --region "$REGION" \
                --instance-ids "$INSTANCE_ID"
            echo "✓ Termination initiated"
            rm "$INFO_FILE"
        else
            echo "Cancelled"
        fi
        ;;
    
    cost)
        RUNNING_TIME=$(aws ec2 describe-instances \
            --region "$REGION" \
            --instance-ids "$INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].LaunchTime' \
            --output text)
        
        CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
        HOURS=$(( ($(date -d "$CURRENT_TIME" +%s) - $(date -d "$RUNNING_TIME" +%s)) / 3600 ))
        
        echo "Instance has been running for ~$HOURS hours"
        echo ""
        echo "Approximate costs:"
        echo "  g4dn.xlarge:  \$$(echo "$HOURS * 0.526" | bc -l | xargs printf "%.2f")"
        echo "  g5.xlarge:    \$$(echo "$HOURS * 1.006" | bc -l | xargs printf "%.2f")"
        echo "  g5.2xlarge:   \$$(echo "$HOURS * 1.212" | bc -l | xargs printf "%.2f")"
        echo "  p3.2xlarge:   \$$(echo "$HOURS * 3.06" | bc -l | xargs printf "%.2f")"
        ;;
    
    help|--help|-h)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  status         Show instance status (default)"
        echo "  ssh            SSH into training instance"
        echo "  logs           Tail training logs"
        echo "  setup-logs     Tail setup logs"
        echo "  download [DIR] Download checkpoints to local directory"
        echo "  cost           Show estimated cost so far"
        echo "  terminate      Terminate the instance"
        echo ""
        echo "Examples:"
        echo "  $0 status"
        echo "  $0 ssh"
        echo "  $0 download ./my-checkpoints"
        echo "  $0 terminate"
        ;;
    
    *)
        echo "Unknown command: $1"
        echo "Run: $0 help"
        exit 1
        ;;
esac
