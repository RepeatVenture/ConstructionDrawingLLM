#!/usr/bin/env python3
"""
Production FastAPI server for LLaVA Construction Drawing Model
Deployed on AWS EC2 with GPU support
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import base64
from PIL import Image
from io import BytesIO
import json
import logging
import torch
import time
import traceback
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/llava-api.log')
    ]
)
logger = logging.getLogger(__name__)

# Global model variables
model = None
processor = None
device = None

# Model configuration
MODEL_CONFIG = {
    "base_model": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "adapter_path": "/home/ubuntu/llava-model/adapters",
    "model_name": "llava-construction-v1",
    "version": "1.0.0",
}


def load_model_on_startup():
    """Load the LLaVA model with trained LoRA adapters at startup."""
    global model, processor, device
    
    try:
        logger.info("=" * 70)
        logger.info("Loading LLaVA Construction Drawing Model")
        logger.info("=" * 70)
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. GPU required for inference.")
        
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Import required libraries
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        from peft import PeftModel
        
        # Load processor
        logger.info(f"Loading processor: {MODEL_CONFIG['base_model']}")
        processor = LlavaNextProcessor.from_pretrained(
            MODEL_CONFIG['base_model'],
            cache_dir="/home/ubuntu/.cache/huggingface"
        )
        
        # Load base model
        logger.info(f"Loading base model: {MODEL_CONFIG['base_model']}")
        base_model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_CONFIG['base_model'],
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="/home/ubuntu/.cache/huggingface"
        )
        
        # Load trained LoRA adapters
        logger.info(f"Loading LoRA adapters: {MODEL_CONFIG['adapter_path']}")
        model = PeftModel.from_pretrained(
            base_model,
            MODEL_CONFIG['adapter_path'],
            torch_dtype=torch.float16
        )
        model.eval()
        
        # Warm up the model
        logger.info("Warming up model with test inference...")
        test_image = Image.new('RGB', (224, 224), color='white')
        test_inputs = processor(
            text="USER: <image>\nTest\nASSISTANT:",
            images=test_image,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            _ = model.generate(**test_inputs, max_new_tokens=10)
        
        torch.cuda.empty_cache()
        
        gpu_memory_used = torch.cuda.memory_allocated(0) / 1e9
        logger.info(f"GPU memory used: {gpu_memory_used:.2f} GB")
        logger.info("✓ Model loaded and ready")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    load_model_on_startup()
    yield
    # Shutdown
    logger.info("Shutting down server...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Initialize FastAPI app
app = FastAPI(
    title="LLaVA Construction Drawing API",
    description="Production API for analyzing construction drawings with custom-trained LLaVA model",
    version=MODEL_CONFIG['version'],
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
        "https://blueprintbid.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class AnalyzeRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image with optional data URI prefix")
    text: str = Field(..., description="User prompt/question about the image")
    system_prompt: Optional[str] = Field("", description="System prompt for context")
    max_tokens: Optional[int] = Field(4000, ge=1, le=8192, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    response_format: Optional[str] = Field("json", description="Expected response format")


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AnalyzeResponse(BaseModel):
    content: str = Field(..., description="JSON string containing analysis results")
    model: str = Field(..., description="Model identifier")
    usage: UsageInfo


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_memory_allocated_gb: Optional[float]
    gpu_memory_total_gb: Optional[float]


class InfoResponse(BaseModel):
    model_name: str
    version: str
    base_model: str
    deployment: str
    adapter_path: str


# Helper Functions
def decode_base64_image(data_url: str) -> Image.Image:
    """Decode base64 image string to PIL Image."""
    try:
        # Handle data URI format (data:image/png;base64,...)
        if ',' in data_url and data_url.startswith('data:'):
            base64_str = data_url.split(',', 1)[1]
        else:
            base64_str = data_url
        
        # Decode base64
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    except Exception as e:
        logger.error(f"Failed to decode image: {str(e)}")
        raise ValueError(f"Invalid base64 image data: {str(e)}")


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token)."""
    return len(text) // 4


def parse_json_from_text(text: str) -> Dict[str, Any]:
    """Extract and parse JSON from model output."""
    # Try to parse the entire text as JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass
    
    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    
    # Return default structure if no valid JSON found
    logger.warning("Could not parse JSON from model output, returning default structure")
    return {
        "included_items": [],
        "specifications": [],
        "_raw_output": text,
        "_parse_error": "Could not extract valid JSON from model output"
    }


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LLaVA Construction Drawing API",
        "version": MODEL_CONFIG['version'],
        "status": "operational",
        "endpoints": {
            "analyze": "POST /analyze - Analyze construction drawings",
            "health": "GET /health - Health check",
            "info": "GET /info - Model information"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    gpu_memory_allocated = None
    gpu_memory_total = None
    
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return HealthResponse(
        status="ok" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_allocated_gb=gpu_memory_allocated,
        gpu_memory_total_gb=gpu_memory_total
    )


@app.get("/info", response_model=InfoResponse)
async def model_info():
    """Model information endpoint."""
    return InfoResponse(
        model_name=MODEL_CONFIG['model_name'],
        version=MODEL_CONFIG['version'],
        base_model=MODEL_CONFIG['base_model'],
        deployment="aws-ec2-gpu",
        adapter_path=MODEL_CONFIG['adapter_path']
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyze construction drawing and extract information.
    
    This endpoint processes a construction drawing image and extracts
    millwork scope items, specifications, and other relevant information.
    """
    start_time = time.time()
    
    try:
        if model is None or processor is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Server is starting up or encountered an error."
            )
        
        logger.info(f"Analyze request received: image_size={len(request.image)} chars, text_len={len(request.text)} chars")
        
        # Decode image
        try:
            image = decode_base64_image(request.image)
            logger.info(f"Image decoded: {image.size} pixels, mode={image.mode}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Prepare prompt
        full_prompt = f"{request.system_prompt}\n\n{request.text}" if request.system_prompt else request.text
        
        # Format in LLaVA conversation format
        conversation = f"USER: <image>\n{full_prompt}\nASSISTANT:"
        
        logger.info(f"Processing with prompt length: {len(conversation)} chars")
        
        # Process inputs
        inputs = processor(
            text=conversation,
            images=image,
            return_tensors="pt"
        ).to(device)
        
        prompt_tokens = inputs['input_ids'].shape[1]
        
        # Generate response
        logger.info(f"Generating response (max_tokens={request.max_tokens})...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        
        # Decode response
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        response_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
        
        completion_tokens = len(generated_ids)
        total_tokens = prompt_tokens + completion_tokens
        
        logger.info(f"Response generated: {len(response_text)} chars, {completion_tokens} tokens")
        
        # Parse JSON from response
        if request.response_format == "json":
            try:
                result_dict = parse_json_from_text(response_text)
                content_str = json.dumps(result_dict)
            except Exception as e:
                logger.error(f"JSON parsing failed: {str(e)}")
                # Return raw response in JSON wrapper
                content_str = json.dumps({
                    "included_items": [],
                    "specifications": [],
                    "_raw_output": response_text,
                    "_parse_error": str(e)
                })
        else:
            content_str = response_text
        
        # Clean up GPU memory
        del inputs, output_ids, generated_ids
        torch.cuda.empty_cache()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Analysis complete in {elapsed_time:.2f}s")
        
        return AnalyzeResponse(
            content=content_str,
            model=MODEL_CONFIG['model_name'],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Clean up GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Production server configuration
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        workers=1,  # Single worker for GPU models
    )
