#!/bin/bash
# ==========================================================================
#  BlueprintBid LLM – Start/Stop EC2 instance to save costs
#
#  Usage:
#    ./manage.sh start    # Start the GPU instance
#    ./manage.sh stop     # Stop (not terminate) – no GPU cost while stopped
#    ./manage.sh status   # Check instance & server status
#    ./manage.sh logs     # Tail container logs via SSM
#    ./manage.sh ssh      # SSH into the instance
# ==========================================================================
set -euo pipefail

STACK_NAME="${STACK_NAME:-blueprintbid-llm}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

get_instance_id() {
    aws cloudformation describe-stacks \
        --region "$REGION" \
        --stack-name "$STACK_NAME" \
        --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" \
        --output text
}

get_endpoint() {
    aws cloudformation describe-stacks \
        --region "$REGION" \
        --stack-name "$STACK_NAME" \
        --query "Stacks[0].Outputs[?OutputKey=='Endpoint'].OutputValue" \
        --output text
}

case "${1:-status}" in
    start)
        INSTANCE_ID=$(get_instance_id)
        echo "Starting instance $INSTANCE_ID ..."
        aws ec2 start-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
        echo "Instance starting. Server will be ready in ~2 minutes."
        echo "Check with: $0 status"
        ;;

    stop)
        INSTANCE_ID=$(get_instance_id)
        echo "Stopping instance $INSTANCE_ID ..."
        echo "  (EBS storage charges continue, but no GPU/compute cost)"
        aws ec2 stop-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
        echo "Instance stopping."
        ;;

    status)
        INSTANCE_ID=$(get_instance_id)
        ENDPOINT=$(get_endpoint)

        echo "Instance: $INSTANCE_ID"
        STATE=$(aws ec2 describe-instances --region "$REGION" \
            --instance-ids "$INSTANCE_ID" \
            --query "Reservations[0].Instances[0].State.Name" --output text)
        echo "State:    $STATE"

        if [ "$STATE" = "running" ]; then
            IP=$(aws ec2 describe-instances --region "$REGION" \
                --instance-ids "$INSTANCE_ID" \
                --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
            echo "IP:       $IP"
            echo ""
            echo "Health check:"
            curl -sf "${ENDPOINT}/health" 2>/dev/null | python3 -m json.tool || echo "  Server not ready yet (model may be loading)"
        fi
        ;;

    logs)
        INSTANCE_ID=$(get_instance_id)
        echo "Connecting via SSM to tail Docker logs ..."
        aws ssm start-session \
            --region "$REGION" \
            --target "$INSTANCE_ID" \
            --document-name AWS-StartInteractiveCommand \
            --parameters command="docker logs -f --tail 50 blueprintbid-llm"
        ;;

    ssh)
        INSTANCE_ID=$(get_instance_id)
        IP=$(aws ec2 describe-instances --region "$REGION" \
            --instance-ids "$INSTANCE_ID" \
            --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
        KEY=$(aws cloudformation describe-stacks \
            --region "$REGION" \
            --stack-name "$STACK_NAME" \
            --query "Stacks[0].Outputs[?OutputKey=='SSHCommand'].OutputValue" --output text)
        echo "Connecting: ssh ubuntu@$IP"
        ssh -o StrictHostKeyChecking=no "ubuntu@$IP"
        ;;

    destroy)
        echo "This will PERMANENTLY delete the stack, instance, and all data."
        read -rp "Type the stack name to confirm: " confirm
        if [ "$confirm" = "$STACK_NAME" ]; then
            aws cloudformation delete-stack --region "$REGION" --stack-name "$STACK_NAME"
            echo "Stack deletion initiated."
        else
            echo "Cancelled."
        fi
        ;;

    *)
        echo "Usage: $0 {start|stop|status|logs|ssh|destroy}"
        exit 1
        ;;
esac
