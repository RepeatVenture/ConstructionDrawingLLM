#!/bin/bash
# ==========================================================================
#  BlueprintBid LLM – EC2 User Data (Bootstrap Script)
#
#  Runs on first boot of an NVIDIA GPU instance (g5.xlarge, g4dn.xlarge, etc.)
#  using the Amazon Linux 2023 or Ubuntu 22.04 Deep Learning AMI.
#
#  This script:
#    1. Installs Docker + NVIDIA Container Toolkit
#    2. Clones the repo (or pulls from ECR)
#    3. Builds the container
#    4. Starts the server on port 8000
# ==========================================================================
set -euo pipefail

LOG="/var/log/blueprintbid-setup.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== BlueprintBid LLM setup started at $(date) ==="

# ---- Config (edit these or pass via CloudFormation Parameters) ----
REPO_URL="${REPO_URL:-https://github.com/RepeatVenture/ConstructionDrawingLLM.git}"
BRANCH="${BRANCH:-main}"
APP_DIR="/opt/blueprintbid"
MODEL_CACHE="/data/model_cache"
QUANTIZATION="${QUANTIZATION:-int4}"
MODEL_BACKEND="${MODEL_BACKEND:-qwen2-vl}"
REQUIRE_AUTH="${REQUIRE_AUTH:-true}"
API_KEY="${API_KEY:-}"                     # set via CloudFormation or SSM
PORT="${PORT:-8000}"

# ---- 1. System updates ----
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get upgrade -y

# ---- 2. Install Docker ----
if ! command -v docker &>/dev/null; then
    apt-get install -y docker.io
    systemctl enable --now docker
    usermod -aG docker ubuntu
fi

# ---- 3. Install NVIDIA Container Toolkit ----
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -y
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# ---- 4. Verify GPU ----
nvidia-smi || { echo "ERROR: nvidia-smi failed. Is this a GPU instance?"; exit 1; }

# ---- 5. Clone repo ----
mkdir -p "$APP_DIR" "$MODEL_CACHE"
if [ ! -d "$APP_DIR/.git" ]; then
    git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
else
    cd "$APP_DIR" && git pull origin "$BRANCH"
fi

# ---- 6. Build and run ----
cd "$APP_DIR"

docker build -t blueprintbid-llm .

docker rm -f blueprintbid-llm 2>/dev/null || true

docker run -d \
    --name blueprintbid-llm \
    --gpus all \
    --restart unless-stopped \
    -p "${PORT}:8000" \
    -v "${MODEL_CACHE}:/app/model_cache" \
    -e MODEL_BACKEND="${MODEL_BACKEND}" \
    -e QUANTIZATION="${QUANTIZATION}" \
    -e REQUIRE_AUTH="${REQUIRE_AUTH}" \
    -e BLUEPRINTBID_API_KEY="${API_KEY}" \
    -e MAX_CONCURRENT_REQUESTS=3 \
    -e ENABLE_OCR=true \
    blueprintbid-llm

# ---- 7. Wait for healthy ----
echo "Waiting for model to load (this takes 1-3 minutes)..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:${PORT}/health >/dev/null 2>&1; then
        echo "=== Server is healthy! ==="
        curl -s http://localhost:${PORT}/health | python3 -m json.tool
        break
    fi
    sleep 5
done

echo "=== BlueprintBid LLM setup completed at $(date) ==="
