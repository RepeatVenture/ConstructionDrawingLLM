#!/bin/bash
set -e

REQUEST_ID="0db2745e4d8e48bfb1bce59156c39e496xPoMG7O"
REGION="us-east-1"
CHECK_INTERVAL=600  # 10 minutes in seconds

echo "========================================================================="
echo "  Monitoring AWS GPU Quota Request"
echo "========================================================================="
echo "  Request ID: $REQUEST_ID"
echo "  Region:     $REGION"
echo "  Checking:   Every 10 minutes"
echo "========================================================================="
echo ""

check_count=0

while true; do
    check_count=$((check_count + 1))
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] Check #$check_count - Querying AWS..."
    
    # Get quota request status
    status_json=$(aws service-quotas get-requested-service-quota-change \
        --request-id "$REQUEST_ID" \
        --region "$REGION" 2>&1)
    
    if echo "$status_json" | grep -q "NoSuchResourceException"; then
        echo "❌ Request not found. It may have been processed already."
        echo ""
        echo "Checking current quota value..."
        current_quota=$(aws service-quotas get-service-quota \
            --service-code ec2 \
            --quota-code L-DB2E81BA \
            --region "$REGION" \
            --query 'Quota.Value' \
            --output text)
        
        if (( $(echo "$current_quota > 0" | bc -l) )); then
            echo "✅ Quota is now $current_quota - Request was approved!"
            break
        else
            echo "⚠️  Quota still at 0. Request may have been denied."
            exit 1
        fi
    fi
    
    status=$(echo "$status_json" | grep -oP '"Status":\s*"\K[^"]+' | head -1)
    
    echo "   Status: $status"
    
    case "$status" in
        CASE_CLOSED)
            echo ""
            echo "✅ APPROVED! Quota request has been granted."
            echo ""
            
            # Verify the new quota value
            new_quota=$(aws service-quotas get-service-quota \
                --service-code ec2 \
                --quota-code L-DB2E81BA \
                --region "$REGION" \
                --query 'Quota.Value' \
                --output text)
            
            echo "   New quota: $new_quota vCPUs"
            echo ""
            break
            ;;
            
        DENIED|CASE_DENIED)
            echo ""
            echo "❌ DENIED: Your quota request was not approved."
            echo ""
            echo "Possible reasons:"
            echo "  - Account too new"
            echo "  - Previous payment issues"
            echo "  - Region-specific restrictions"
            echo ""
            echo "You can:"
            echo "  1. Try requesting in a different region (us-west-2, eu-west-1)"
            echo "  2. Contact AWS support for manual review"
            echo "  3. Use Modal instead (no quota needed)"
            echo ""
            exit 1
            ;;
            
        PENDING|CASE_OPENED)
            echo "   ⏳ Still pending... (Total wait: $((check_count * 10)) minutes)"
            if [ $check_count -eq 1 ]; then
                echo ""
                echo "   AWS typically approves GPU quota requests within 15-60 minutes."
                echo "   I'll keep checking automatically."
            fi
            echo ""
            ;;
            
        *)
            echo "   ⚠️  Unknown status: $status"
            echo ""
            ;;
    esac
    
    # If still pending, wait before next check
    if [ "$status" = "PENDING" ] || [ "$status" = "CASE_OPENED" ]; then
        echo "   Next check in 10 minutes at $(date -d "+10 minutes" '+%H:%M:%S')..."
        echo ""
        sleep $CHECK_INTERVAL
    fi
done

echo "========================================================================="
echo "  Launching Training Instance"
echo "========================================================================="
echo ""

# Launch the training instance
cd /workspaces/ConstructionDrawingLLM/training
./aws_train.sh --key construction-llm

echo ""
echo "========================================================================="
echo "✅ Training instance launched successfully!"
echo "========================================================================="
echo ""
echo "Next steps:"
echo "  1. Wait ~15-20 minutes for setup to complete"
echo "  2. SSH into instance: ./aws_manage.sh ssh"
echo "  3. Start training: ./start_training.sh"
echo "  4. Monitor: ./aws_manage.sh logs"
echo "  5. When done: ./aws_manage.sh terminate"
echo ""
