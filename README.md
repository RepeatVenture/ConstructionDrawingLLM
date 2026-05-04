# BlueprintBid Local LLM – Construction Drawing Analysis

A production-ready local LLM server that replaces OpenAI GPT-4 Vision for analyzing construction drawings and extracting scope items (quantities for bidding).

## Architecture

```
                          ┌───────────────────────────┐
                          │   BlueprintBid Next.js     │
                          │   (existing SaaS app)      │
                          └──────────┬────────────────┘
                                     │ POST /analyze-drawing
                                     ▼
                          ┌───────────────────────────┐
                          │   FastAPI Server           │
                          │   (this project)           │
                          ├───────────────────────────┤
                          │  Image Preprocessing       │
                          │  OCR (EasyOCR)             │
                          │  Trade-specific Prompts    │
                          │  Qwen2.5-VL Inference      │
                          │  Output Parsing & Validate │
                          └───────────────────────────┘
                                     │
                          ┌──────────┴────────────────┐
                          │  GPU (local or cloud)      │
                          │  RTX 3090/4090 or A10G     │
                          └───────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install (GPU version)
pip install -r requirements.txt
```

### 2. Download Model Weights

```bash
python scripts/download_model.py
# Downloads Qwen2.5-VL-7B-Instruct (~14GB) to ./model_cache/
```

### 3. Start the Server

```bash
python server.py
# Server starts at http://localhost:8000
```

### 4. Test It

```bash
# Health check
curl http://localhost:8000/health

# Analyze a drawing
curl -X POST http://localhost:8000/analyze-drawing \
  -F "image=@page.png" \
  -F "trade=millwork"

# Or run the smoke test
bash scripts/test_api.sh
```

## API Reference

### `POST /analyze-drawing` (multipart/form-data)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | ✅ | PNG image of a construction drawing page |
| `text` | string | ❌ | OCR-extracted text (auto-extracted if empty) |
| `trade` | string | ❌ | `millwork` \| `plumbing` \| `electrical` \| `flooring` (default: `millwork`) |

### `POST /analyze-drawing-json` (application/json)

```json
{
  "image_base64": "<base64-encoded PNG>",
  "text": "",
  "trade": "millwork"
}
```

### `GET /health`

Returns server status, loaded model, GPU info.

### Response Format

```json
{
  "scopeItems": [
    {
      "id": "item-a1b2c3d4",
      "category": "Cabinetry",
      "description": "Wall cabinet 17 l/f",
      "quantity": 17,
      "unit": "lf",
      "confidence": 0.95,
      "pageRefs": [1],
      "textSnippets": ["Wall cabinet 17 LF"]
    }
  ],
  "specSheet": [
    {
      "key": "Cabinet Finish",
      "value": "PL-1",
      "confidence": 0.9,
      "pageRefs": [1],
      "textSnippets": ["Finish: PL-1"]
    }
  ],
  "specifications": [
    {
      "finish_code": "PL-1",
      "finish_name": "Plastic Laminate",
      "category": "Casework",
      "manufacturer": "Wilsonart",
      "description": "High pressure laminate",
      "finish": "Matte"
    }
  ]
}
```

## Configuration

All settings via environment variables (see [.env.example](.env.example)):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_BACKEND` | `qwen2-vl` | `qwen2-vl` \| `llava` \| `internvl2` |
| `QUANTIZATION` | `int4` | `none` \| `int8` \| `int4` \| `gptq` \| `awq` |
| `MAX_IMAGE_DIM` | `2048` | Max image dimension (longest side) |
| `ENABLE_OCR` | `true` | Run OCR on images automatically |
| `MAX_CONCURRENT_REQUESTS` | `3` | Concurrent inference limit |
| `REQUIRE_AUTH` | `false` | Require Bearer token |
| `BLUEPRINTBID_API_KEY` | | API key (when auth enabled) |
| `PORT` | `8000` | Server port |
| `TEMPERATURE` | `0.1` | LLM temperature (lower = more deterministic) |

## Model Options

| Model | VRAM (INT4) | VRAM (FP16) | Best For |
|-------|-------------|-------------|----------|
| **Qwen2.5-VL-7B** (default) | ~6 GB | ~14 GB | Best document/diagram understanding |
| **Qwen2.5-VL-3B** | ~3 GB | ~6 GB | Lower VRAM, decent accuracy |
| **LLaVA-NeXT-7B** | ~6 GB | ~14 GB | Good general vision, easy to fine-tune |
| **InternVL2-8B** | ~6 GB | ~16 GB | Strong document OCR |

To use a different model:
```bash
export MODEL_BACKEND=llava
# or for a specific model:
export QWEN2_VL_MODEL_ID=Qwen/Qwen2.5-VL-3B
```

## Deployment

### Option 1: Docker (Local/Cloud GPU)

```bash
# Build
docker build -t blueprintbid-llm .

# Run with GPU
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/model_cache:/app/model_cache \
  blueprintbid-llm

# Or use docker compose
docker compose up --build
```

### Option 2: Modal Serverless GPU (Recommended for Production)

```bash
pip install modal
modal setup  # one-time auth

# Download model to Modal volume
modal run modal_deploy.py::download_model

# Deploy
modal deploy modal_deploy.py

# Your endpoint:
# https://<your-username>--blueprintbid-inference-analyze-drawing-endpoint.modal.run
```

### Option 3: AWS ECS

See Dockerfile. Deploy to ECS with GPU task definition (g5.xlarge).

## Integration with BlueprintBid

Copy [integration/local-llm-provider.ts](integration/local-llm-provider.ts) to your BlueprintBid project:

```bash
cp integration/local-llm-provider.ts /path/to/blueprintbid/lib/document-ai/local-llm-provider.ts
```

Then in your extraction API route:

```typescript
// app/api/documents/extract/route.ts
import { extractWithLocalLLM } from '@/lib/document-ai/local-llm-provider';
// or for production:
import { extractWithHybrid } from '@/lib/document-ai/local-llm-provider';

// Replace the OpenAI call:
const result = await extractWithLocalLLM(imageBuffer, ocrText, trade);
```

Environment variables in BlueprintBid:
```env
LOCAL_LLM_URL=http://localhost:8000
# or for Modal:
MODAL_LLM_URL=https://your--blueprintbid-inference-analyze-drawing-endpoint.modal.run
MODAL_TOKEN=your-modal-token
```

## Project Structure

```
ConstructionDrawingLLM/
├── server.py                   # FastAPI server (main entry)
├── config.py                   # All configuration
├── modal_deploy.py             # Modal serverless deployment
├── Dockerfile                  # GPU Docker image
├── docker-compose.yml          # Docker compose with GPU
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
│
├── models/                     # VLM providers
│   ├── base.py                 # Abstract provider interface
│   ├── qwen2_vl.py             # Qwen2.5-VL (recommended)
│   ├── llava_provider.py       # LLaVA-NeXT alternative
│   └── provider_factory.py     # Provider singleton factory
│
├── prompts/                    # Trade-specific prompts
│   └── trade_prompts.py        # Millwork, plumbing, electrical, flooring
│
├── processing/                 # Input/output processing
│   ├── image_processor.py      # Image load, resize, enhance
│   ├── ocr_provider.py         # EasyOCR / PaddleOCR
│   └── output_parser.py        # JSON parse, validate, normalize
│
├── schemas/                    # Pydantic models
│   └── extraction.py           # ExtractionResult, ScopeItem, etc.
│
├── integration/                # BlueprintBid client code
│   └── local-llm-provider.ts   # TypeScript drop-in for Next.js
│
├── scripts/                    # Utilities
│   ├── download_model.py       # Download model weights
│   ├── benchmark.py            # Performance benchmarking
│   └── test_api.sh             # API smoke tests
│
└── tests/                      # Test suite
    ├── test_output_parser.py   # Parser tests (no GPU needed)
    ├── test_image_processor.py # Image processing tests
    ├── test_schemas.py         # Schema validation tests
    ├── test_prompts.py         # Prompt builder tests
    └── test_server.py          # API endpoint tests (mocked)
```

## Testing

```bash
# Run all tests (no GPU required)
pip install pytest httpx
pytest

# Run specific test file
pytest tests/test_output_parser.py -v

# Run with coverage
pip install pytest-cov
pytest --cov=. --cov-report=html
```

## Benchmarking

```bash
# Single image
python scripts/benchmark.py --image path/to/page.png --trade millwork

# Directory of images
python scripts/benchmark.py --image-dir tests/sample_images/ --trade millwork

# Multiple passes
python scripts/benchmark.py --image-dir tests/sample_images/ --repeat 3
```

## Performance Tuning

### Speed
- Use `QUANTIZATION=int4` (fastest, minimal accuracy loss)
- Use `MAX_IMAGE_DIM=1536` for faster processing
- Set `ENABLE_OCR=false` if providing text externally
- Use `TEMPERATURE=0.0` for deterministic output

### Accuracy
- Use `QUANTIZATION=none` or `int8` for best accuracy
- Use `MAX_IMAGE_DIM=2048` or higher
- Keep `ENABLE_OCR=true` for supplementary text context
- Consider fine-tuning on your labeled construction drawings

### VRAM
- `int4`: ~6 GB for 7B model (fits RTX 3060)
- `int8`: ~8 GB (fits RTX 3070)
- `none` (FP16): ~14 GB (needs RTX 3090/4090)

## Fine-tuning Guide

To improve accuracy on your specific construction drawings:

1. **Prepare training data**: Label 50-200 construction drawing pages with expected JSON output
2. **Format for training**: Each sample = (image, system_prompt, user_prompt, expected_json)
3. **Fine-tune with LoRA** (parameter-efficient, fits on single GPU)
4. **Evaluate**: Run benchmark against held-out test set
5. **Deploy**: Update `QWEN2_VL_MODEL_ID` to point to your fine-tuned checkpoint

## Cost Comparison

| Solution | Cost/Page | 10K Pages/Mo | Notes |
|----------|-----------|-------------|-------|
| OpenAI GPT-4V | $0.020 | $200 | Reliable, no setup |
| This project (Modal) | $0.007 | $70 | Auto-scales, 65% cheaper |
| This project (AWS g5) | $0.004 | $40* | *Plus $432/mo base |
| This project (local 4090) | ~$0.001 | ~$10* | *Plus hardware cost |

## License

Private – BlueprintBid proprietary.