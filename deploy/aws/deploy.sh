#!/bin/bash
# ==========================================================================
#  BlueprintBid LLM – AWS Quick Deploy Script
#
#  Deploys a single GPU EC2 instance via CloudFormation.
#
#  Prerequisites:
#    - AWS CLI configured (aws configure)
#    - An EC2 key pair
#    - A VPC with at least 2 subnets
#
#  Usage:
#    ./deploy.sh                              # interactive prompts
#    ./deploy.sh --key my-key --region us-east-1
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STACK_NAME="blueprintbid-llm"

# ---- Defaults ----
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
INSTANCE_TYPE="g5.xlarge"
QUANTIZATION="int4"
KEY_PAIR=""
VPC_ID=""
SUBNET_IDS=""
CERT_ARN=""

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --key)          KEY_PAIR="$2";       shift 2 ;;
        --region)       REGION="$2";         shift 2 ;;
        --instance)     INSTANCE_TYPE="$2";  shift 2 ;;
        --vpc)          VPC_ID="$2";         shift 2 ;;
        --subnets)      SUBNET_IDS="$2";     shift 2 ;;
        --cert)         CERT_ARN="$2";       shift 2 ;;
        --quantization) QUANTIZATION="$2";   shift 2 ;;
        --stack-name)   STACK_NAME="$2";     shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --key NAME         EC2 key pair name (required)"
            echo "  --region REGION    AWS region (default: us-east-1)"
            echo "  --instance TYPE    Instance type (default: g5.xlarge)"
            echo "  --vpc ID           VPC ID (auto-detected if omitted)"
            echo "  --subnets IDs      Comma-separated subnet IDs (auto-detected)"
            echo "  --cert ARN         ACM certificate ARN for HTTPS"
            echo "  --quantization Q   int4, int8, or none (default: int4)"
            echo "  --stack-name NAME  CloudFormation stack name"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Verify AWS CLI ----
if ! command -v aws &>/dev/null; then
    echo "ERROR: AWS CLI not found. Install from https://aws.amazon.com/cli/"
    exit 1
fi

echo "Region: $REGION"
aws sts get-caller-identity --region "$REGION" --output table

# ---- Auto-detect VPC ----
if [ -z "$VPC_ID" ]; then
    VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" \
        --filters "Name=is-default,Values=true" \
        --query "Vpcs[0].VpcId" --output text)
    if [ "$VPC_ID" = "None" ] || [ -z "$VPC_ID" ]; then
        echo "ERROR: No default VPC found. Specify --vpc <vpc-id>"
        exit 1
    fi
    echo "Using default VPC: $VPC_ID"
fi

# ---- Auto-detect subnets ----
if [ -z "$SUBNET_IDS" ]; then
    SUBNET_IDS=$(aws ec2 describe-subnets --region "$REGION" \
        --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
        --query "Subnets[*].SubnetId" --output text | tr '\t' ',')
    echo "Using subnets: $SUBNET_IDS"
fi

# ---- Key pair check ----
if [ -z "$KEY_PAIR" ]; then
    echo ""
    echo "Available key pairs in $REGION:"
    aws ec2 describe-key-pairs --region "$REGION" \
        --query "KeyPairs[*].KeyName" --output table
    echo ""
    read -rp "Enter key pair name: " KEY_PAIR
fi

# ---- Generate API key ----
API_KEY=$(openssl rand -hex 24)
echo ""
echo "Generated API key (save this!):"
echo "  $API_KEY"
echo ""

# ---- Cost warning ----
echo "========================================"
echo "  Instance: $INSTANCE_TYPE"
case "$INSTANCE_TYPE" in
    g5.xlarge)    echo "  Cost: ~\$1.01/hr (\$730/mo on-demand)" ;;
    g5.2xlarge)   echo "  Cost: ~\$1.52/hr (\$1,094/mo on-demand)" ;;
    g4dn.xlarge)  echo "  Cost: ~\$0.53/hr (\$380/mo on-demand)" ;;
    g4dn.2xlarge) echo "  Cost: ~\$0.75/hr (\$540/mo on-demand)" ;;
    p3.2xlarge)   echo "  Cost: ~\$3.06/hr (\$2,203/mo on-demand)" ;;
esac
echo "  Tip: Use Savings Plans or stop when idle"
echo "========================================"
echo ""
read -rp "Continue? (y/N) " confirm
[[ "$confirm" =~ ^[Yy] ]] || exit 0

# ---- Deploy CloudFormation ----
echo ""
echo "Deploying CloudFormation stack: $STACK_NAME ..."

aws cloudformation deploy \
    --region "$REGION" \
    --stack-name "$STACK_NAME" \
    --template-file "$SCRIPT_DIR/cloudformation.yml" \
    --parameter-overrides \
        InstanceType="$INSTANCE_TYPE" \
        KeyPairName="$KEY_PAIR" \
        ApiKey="$API_KEY" \
        VpcId="$VPC_ID" \
        SubnetIds="$SUBNET_IDS" \
        CertificateArn="${CERT_ARN}" \
        Quantization="$QUANTIZATION" \
    --capabilities CAPABILITY_IAM \
    --no-fail-on-empty-changeset

# ---- Get outputs ----
echo ""
echo "=== Deployment Complete ==="
aws cloudformation describe-stacks \
    --region "$REGION" \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs" \
    --output table

ENDPOINT=$(aws cloudformation describe-stacks \
    --region "$REGION" \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='Endpoint'].OutputValue" \
    --output text)

echo ""
echo "=== Quick Test ==="
echo ""
echo "# Health check:"
echo "curl -s ${ENDPOINT}/health | python3 -m json.tool"
echo ""
echo "# Analyze a drawing:"
echo "curl -X POST ${ENDPOINT}/analyze-drawing \\"
echo "  -H 'Authorization: Bearer ${API_KEY}' \\"
echo "  -F 'image=@drawing.png' \\"
echo "  -F 'trade=millwork'"
echo ""
echo "# In your BlueprintBid .env:"
echo "LOCAL_LLM_URL=${ENDPOINT}"
echo "LOCAL_LLM_API_KEY=${API_KEY}"
echo ""
echo "NOTE: The server needs 2-5 minutes to download and load the model on first boot."
echo "      Monitor with: aws ssm start-session --target <instance-id>"
echo "      Or SSH and run: docker logs -f blueprintbid-llm"
