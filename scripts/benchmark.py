#!/usr/bin/env python3
"""
Benchmark the LLM server against sample images.

Usage:
    python scripts/benchmark.py --url http://localhost:8000 --image-dir tests/sample_images/
    python scripts/benchmark.py --url http://localhost:8000 --image path/to/page.png

Reports:
- Latency (p50, p95, p99)
- Items extracted per page
- Confidence distribution
- Error rate
"""
import argparse
import base64
import json
import os
import sys
import statistics
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)


def analyze_image(url: str, image_path: str, trade: str, api_key: str = "") -> dict:
    """Send an image to the server and return timing + result."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.perf_counter()
    resp = requests.post(
        f"{url}/analyze-drawing",
        files={"image": (os.path.basename(image_path), image_bytes, "image/png")},
        data={"trade": trade, "text": ""},
        headers=headers,
        timeout=120,
    )
    elapsed = time.perf_counter() - t0

    return {
        "path": image_path,
        "status": resp.status_code,
        "elapsed": elapsed,
        "result": resp.json() if resp.status_code == 200 else None,
        "error": resp.text if resp.status_code != 200 else None,
    }


def print_report(results: list):
    """Print a summary report."""
    successes = [r for r in results if r["status"] == 200]
    failures = [r for r in results if r["status"] != 200]

    print("\n" + "=" * 60)
    print("BENCHMARK REPORT")
    print("=" * 60)

    print(f"\nTotal images: {len(results)}")
    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")
    print(f"Error rate: {len(failures) / len(results) * 100:.1f}%")

    if successes:
        times = [r["elapsed"] for r in successes]
        times.sort()
        print(f"\nLatency:")
        print(f"  Min:  {min(times):.1f}s")
        print(f"  p50:  {statistics.median(times):.1f}s")
        if len(times) >= 20:
            p95_idx = int(len(times) * 0.95)
            p99_idx = int(len(times) * 0.99)
            print(f"  p95:  {times[p95_idx]:.1f}s")
            print(f"  p99:  {times[p99_idx]:.1f}s")
        print(f"  Max:  {max(times):.1f}s")
        print(f"  Mean: {statistics.mean(times):.1f}s")

        item_counts = []
        confidences = []
        for r in successes:
            items = r["result"].get("scopeItems", [])
            item_counts.append(len(items))
            for item in items:
                confidences.append(item.get("confidence", 0))

        print(f"\nScope items per page:")
        print(f"  Min:  {min(item_counts)}")
        print(f"  Max:  {max(item_counts)}")
        print(f"  Mean: {statistics.mean(item_counts):.1f}")

        if confidences:
            print(f"\nConfidence scores:")
            print(f"  Min:  {min(confidences):.2f}")
            print(f"  Mean: {statistics.mean(confidences):.2f}")
            print(f"  Max:  {max(confidences):.2f}")

    if failures:
        print(f"\nFailures:")
        for r in failures:
            print(f"  {r['path']}: HTTP {r['status']} – {r['error'][:100]}")

    print()

    # Per-image details
    print("Per-image details:")
    print(f"{'Image':<40} {'Time':>6} {'Items':>5} {'Status':>6}")
    print("-" * 60)
    for r in results:
        name = os.path.basename(r["path"])[:38]
        items = len(r["result"]["scopeItems"]) if r["result"] else 0
        print(f"{name:<40} {r['elapsed']:>5.1f}s {items:>5} {r['status']:>6}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the LLM server")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--image", type=str, help="Single image to test")
    parser.add_argument("--image-dir", type=str, help="Directory of images to test")
    parser.add_argument("--trade", default="millwork", help="Trade to analyze")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--repeat", type=int, default=1, help="Times to repeat each image")
    args = parser.parse_args()

    # Collect images
    images = []
    if args.image:
        images.append(args.image)
    elif args.image_dir:
        d = Path(args.image_dir)
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tiff"):
            images.extend(str(p) for p in d.glob(ext))
        images.sort()
    else:
        print("Provide --image or --image-dir")
        sys.exit(1)

    if not images:
        print("No images found")
        sys.exit(1)

    print(f"Benchmarking {len(images)} images × {args.repeat} repeats = {len(images) * args.repeat} requests")
    print(f"Server: {args.url}")
    print(f"Trade: {args.trade}")

    # Check server health
    try:
        health = requests.get(f"{args.url}/health", timeout=5)
        print(f"Server health: {health.json()}")
    except Exception as e:
        print(f"Warning: health check failed: {e}")

    # Run benchmark
    results = []
    total = len(images) * args.repeat
    for rep in range(args.repeat):
        for i, img_path in enumerate(images):
            idx = rep * len(images) + i + 1
            print(f"  [{idx}/{total}] {os.path.basename(img_path)} … ", end="", flush=True)
            result = analyze_image(args.url, img_path, args.trade, args.api_key)
            print(f"{result['elapsed']:.1f}s  ({result['status']})")
            results.append(result)

    print_report(results)


if __name__ == "__main__":
    main()
