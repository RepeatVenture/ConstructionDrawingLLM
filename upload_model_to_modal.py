"""
Upload trained model adapters to Modal Volume

This script uploads your trained LoRA adapters to Modal's persistent storage
so they can be loaded by the serverless inference endpoint.

Usage:
    modal run upload_model_to_modal.py
"""

import modal
from pathlib import Path

app = modal.App("upload-model")

# Create image with local model files embedded
local_model_path = Path(__file__).parent / "trained_model" / "final"
image = modal.Image.debian_slim().add_local_dir(local_model_path, remote_path="/tmp/model_files")

@app.function(
    image=image,
    volumes={"/root/adapters": modal.Volume.from_name("construction-model-adapters", create_if_missing=True)},
    timeout=600,
)
def upload_adapters():
    """Upload trained model adapters to Modal volume"""
    import shutil
    
    print("=" * 70)
    print("Uploading Trained Model Adapters to Modal")
    print("=" * 70)
    
    # Source: embedded model files
    src_dir = Path("/tmp/model_files")
    
    # Destination: Modal volume
    dst_dir = Path("/root/adapters")
    
    # Check source exists
    if not src_dir.exists():
        print(f"❌ Error: Source directory not found: {src_dir}")
        return False
    
    # List files to upload
    files = list(src_dir.glob("*"))
    print(f"\nFound {len(files)} files to upload:")
    for f in files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name} ({size_mb:.2f} MB)")
    
    # Copy files
    print(f"\nCopying to Modal volume: {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    for src_file in files:
        dst_file = dst_dir / src_file.name
        print(f"  Copying {src_file.name}...", end=" ")
        shutil.copy2(src_file, dst_file)
        print("✓")
    
    # Verify
    print("\nVerifying upload...")
    uploaded_files = list(dst_dir.glob("*"))
    print(f"✓ {len(uploaded_files)} files in Modal volume")
    
    # Check for required files
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required_files if not (dst_dir / f).exists()]
    
    if missing:
        print(f"⚠️  Warning: Missing required files: {missing}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ Model adapters uploaded successfully!")
    print("=" * 70)
    print("\nNext step: Deploy the API with:")
    print("  modal deploy modal_serve.py")
    
    return True

@app.local_entrypoint()
def main():
    """Upload model adapters"""
    success = upload_adapters.remote()
    if success:
        print("\n✓ Upload complete!")
    else:
        print("\n❌ Upload failed!")
