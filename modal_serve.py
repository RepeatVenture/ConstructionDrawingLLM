"""
Modal Serverless Deployment for LLaVA Construction Drawing Model

This deploys your trained LLaVA model as a serverless web endpoint.
Only charges when GPU is actively processing requests - scales to zero when idle!

Usage:
    modal deploy modal_serve.py
    
Then call the endpoint:
    https://YOUR_APP--inference.modal.run/analyze
"""

import modal
import base64
import json
import io
import re
from pathlib import Path

# ============================================================================
# Modal Configuration
# ============================================================================

# Create Modal app
app = modal.App("llava-construction-api")

# GPU configuration - choose one:
GPU_CONFIG = "T4"  # Cheaper: $0.00060/sec (~$0.05 per inference)
# GPU_CONFIG = "A10G"  # Faster: $0.00136/sec (~$0.11 per inference)

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "numpy<2",  # NumPy 1.x for compatibility
        "transformers==4.46.0",
        "peft==0.13.0",
        "accelerate==0.28.0",
        "bitsandbytes==0.43.0",
        "pillow==10.4.0",
        "fastapi[standard]==0.115.0",
        "sentencepiece",  # Required for LLaMA tokenizer
        "protobuf",
    )
)

# Mount trained model adapters
ADAPTER_PATH = Path(__file__).parent / "trained_model" / "final"
model_volume = modal.Volume.from_name("construction-checkpoints", create_if_missing=False)

# ============================================================================
# Model Loading
# ============================================================================

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    scaledown_window=300,  # Keep warm for 5 minutes after last request
    timeout=600,  # 10 minute timeout for long requests
    secrets=[],  # Add secrets if needed
)
class LLaVAInference:
    """Serverless LLaVA inference class"""
    
    @modal.enter()
    def load_model(self):
        """Load model when container starts"""
        import torch
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        from peft import PeftModel
        
        print("=" * 70)
        print("Loading LLaVA Construction Drawing Model")
        print("=" * 70)
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Load processor
        print("Loading processor...")
        self.processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-vicuna-7b-hf"
        )
        
        # Load base model
        print("Loading base model...")
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-vicuna-7b-hf",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Load LoRA adapters from local directory
        adapter_path = "/root/adapters"
        print(f"Loading LoRA adapters from {adapter_path}...")
        
        try:
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                torch_dtype=torch.float16,
            )
            print("✓ LoRA adapters loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load LoRA adapters: {e}")
            print("⚠️  Using base model without fine-tuning")
        
        self.model.eval()
        print("✓ Model loaded and ready")
        print("=" * 70)
    
    def parse_json_from_text(self, text: str) -> dict:
        """Extract JSON from model output"""
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Return default structure if parsing fails
        return {
            "included_items": [],
            "specifications": [],
            "notes": text[:500] if text else "No response generated"
        }
    
    @modal.method()
    def generate(
        self,
        image_b64: str,
        text: str,
        system_prompt: str = "",
        max_tokens: int = 4000,
        temperature: float = 0.7,
    ) -> dict:
        """Generate response for an image and text prompt"""
        import torch
        from PIL import Image
        
        try:
            # Decode image
            if image_b64.startswith('data:image'):
                image_b64 = image_b64.split(',', 1)[1]
            
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Format prompt for LLaVA
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{text}"
            else:
                full_prompt = text
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": full_prompt},
                    ],
                },
            ]
            
            # Process inputs
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            
            # Decode output
            output_text = self.processor.decode(
                output_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Parse JSON from output
            parsed_json = self.parse_json_from_text(output_text)
            
            # Calculate token counts (approximate)
            prompt_tokens = inputs['input_ids'].shape[1]
            completion_tokens = output_ids.shape[1] - prompt_tokens
            
            # Clean up GPU memory
            del inputs, output_ids
            torch.cuda.empty_cache()
            
            return {
                "content": json.dumps(parsed_json),
                "raw_output": output_text,
                "model": "llava-construction-v1",
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "content": json.dumps({
                    "included_items": [],
                    "specifications": [],
                    "error": str(e)
                })
            }

# ============================================================================
# Web Endpoints (FastAPI)
# ============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

web_app = FastAPI(title="LLaVA Construction API")

# Add CORS
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    image: str
    text: str
    system_prompt: Optional[str] = ""
    max_tokens: Optional[int] = 4000
    temperature: Optional[float] = 0.7
    response_format: Optional[str] = "json"

@web_app.get("/")
def root():
    return {"message": "LLaVA Construction API", "version": "1.0.0"}

@web_app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "llava-construction-v1",
        "version": "1.0.0"
    }

@web_app.get("/info")
def info():
    """Model info endpoint"""
    return {
        "model": "llava-construction-v1",
        "base_model": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "training": {
            "method": "LoRA fine-tuning",
            "dataset": "Construction drawing millwork extraction",
            "samples": 519,
            "epochs": 3,
        },
        "version": "1.0.0",
        "gpu": str(GPU_CONFIG),
    }

@web_app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Main inference endpoint
    
    Request body:
    {
        "image": "data:image/png;base64,..." or base64 string,
        "text": "Extract millwork items...",
        "system_prompt": "You are an expert...",
        "max_tokens": 4000,
        "temperature": 0.7,
        "response_format": "json"
    }
    
    Response:
    {
        "content": "JSON string",
        "model": "llava-construction-v1",
        "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    }
    """
    # Run inference using Modal class
    result = await LLaVAInference().generate.remote.aio(
        image_b64=request.image,
        text=request.text,
        system_prompt=request.system_prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    
    return result

# Mount FastAPI app
@app.function(
    image=image,
    volumes={"/root/adapters": modal.Volume.from_name("construction-model-adapters", create_if_missing=True)},
    timeout=900,  # 15 minute timeout for cold starts (downloading base model on first run)
)
@modal.asgi_app()
def fastapi_app():
    return web_app

# ============================================================================
# Local Testing
# ============================================================================

@app.local_entrypoint()
def test():
    """Test the deployment locally"""
    import base64
    from pathlib import Path
    
    # Find a test image
    test_images = list(Path("training/data/raw").rglob("*.png"))
    if not test_images:
        print("No test images found!")
        return
    
    test_image = test_images[0]
    print(f"Testing with: {test_image}")
    
    # Read and encode image
    with open(test_image, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    
    # Test inference
    inference = LLaVAInference()
    result = inference.generate.remote(
        image_b64=image_b64,
        text="Extract all millwork items from this construction drawing.",
        system_prompt="You are an expert at reading construction drawings.",
        max_tokens=2000,
        temperature=0.7,
    )
    
    print("\n" + "=" * 70)
    print("RESULT:")
    print("=" * 70)
    print(json.dumps(result, indent=2))
