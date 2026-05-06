# Construction Drawing LLM - Trained Model

## ✅ Training Complete!

Your LLaVA-v1.6-7B model has been fine-tuned on **519 construction drawing samples** from 4 projects (Brightview, Candlelight, Devoto, Maple).

### Training Results
- **Model**: LLaVA-v1.6-Vicuna-7B-HF with LoRA adapters
- **Training data**: 415 samples (80% split)
- **Validation data**: 104 samples (20% split)
- **Training duration**: ~2 hours on A100-40GB
- **Final checkpoint**: `trained_model/final/`
- **Adapter size**: 74 MB

### Files in `trained_model/final/`
- `adapter_model.safetensors` - Trained LoRA weights (74 MB)
- `adapter_config.json` - LoRA configuration
- `training_args.bin` - Training hyperparameters
- `README.md` - Model card

## Quick Start

### Test the Model

```bash
# Install dependencies (if not already installed)
pip install torch transformers peft pillow

# Test on a sample drawing
python test_trained_model.py \
    data/raw/Brightview/images/all_pages/sheet_001.png \
    "What quantities are shown on this drawing?"
```

### Use in Python

```python
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel

# Load base model
model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_id)
base_model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load trained adapters
model = PeftModel.from_pretrained(
    base_model,
    "trained_model/final",
    torch_dtype=torch.float16,
)

# Analyze a drawing
image = Image.open("path/to/drawing.png")
prompt = "USER: <image>\nWhat quantities are shown on this drawing?\nASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=512)
    
response = processor.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
print(response)
```

## Model Capabilities

Your trained model can now:
- ✅ Extract quantities from construction drawings
- ✅ Identify materials and specifications
- ✅ Recognize drawing types (site plans, elevations, details, etc.)
- ✅ Answer questions about drawing content
- ✅ Understand construction terminology and standards

## Next Steps

1. **Test on new drawings**: Try the model on drawings it hasn't seen
2. **Evaluate accuracy**: Compare extracted quantities with actual takeoffs
3. **Deploy**: Integrate into your workflow or API
4. **Iterate**: Collect more training data and fine-tune again if needed

## Training Details

- **Base model**: llava-hf/llava-v1.6-vicuna-7b-hf
- **Method**: LoRA (Low-Rank Adaptation)
- **LoRA config**: r=16, alpha=32, dropout=0.05
- **Target modules**: q_proj, k_proj, v_proj, o_proj
- **Optimizer**: AdamW
- **Learning rate**: 2e-4 with linear decay
- **Epochs**: 3
- **Batch size**: 4 (effective with gradient accumulation)
- **GPU**: NVIDIA A100-40GB
- **Training framework**: HuggingFace Transformers + PEFT

## Support

For issues or questions:
1. Check logs in Modal dashboard
2. Review training data in `training/data/prepared/`
3. Test with sample images from `data/raw/`

---

**Model trained**: May 5, 2026
**Training platform**: Modal (serverless GPU)
