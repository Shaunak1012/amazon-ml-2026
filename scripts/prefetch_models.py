"""Pre-download likely backbones into the Hugging Face cache so Day-1 runs start instantly.

    python scripts/prefetch_models.py --dry-run     # list models + download size, fetch nothing
    python scripts/prefetch_models.py               # download (resumable; skips what's cached)
    python scripts/prefetch_models.py --only e5-base siglip-base

Only weights in safetensors (or .bin when there is no safetensors), configs and tokenizer files are fetched,
not duplicate ONNX/TF/Flax copies. Check each model's licence and size against docs/COMPETITION.md before
using it in a submission.
"""
from __future__ import annotations

import argparse
import sys

# short name -> (repo id, licence, why)
MODELS = {
    "e5-base": ("intfloat/e5-base-v2", "MIT", "strong general text embeddings"),
    "bge-base": ("BAAI/bge-base-en-v1.5", "MIT", "text embeddings, second opinion for ensembling"),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "Apache-2.0", "tiny/fast text embeddings for quick baselines"),
    "deberta-v3-base": ("microsoft/deberta-v3-base", "MIT", "best-in-class text fine-tuning backbone"),
    "siglip-base": ("google/siglip-base-patch16-224", "Apache-2.0", "image/text embeddings (strong, fast)"),
    "clip-vit-l14": ("openai/clip-vit-large-patch14", "MIT", "classic strong image/text embeddings"),
}
WEIGHT_PATTERNS = ["*.json", "*.txt", "*.model", "*.spm", "tokenizer*", "vocab*", "merges*", "*.py"]


def plan(repo: str):
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata=True)
    files = {s.rfilename: (s.size or 0) for s in info.siblings}
    has_st = any(f.endswith(".safetensors") for f in files)
    weights = [f for f in files if (f.endswith(".safetensors") if has_st else f.endswith(".bin"))
               and "/" not in f]  # skip onnx/openvino subfolders
    import fnmatch

    extra = [f for f in files if "/" not in f and any(fnmatch.fnmatch(f, p) for p in WEIGHT_PATTERNS)]
    allow = sorted(set(weights + extra))
    return allow, sum(files[f] for f in allow)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(MODELS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    from huggingface_hub import snapshot_download

    total = 0
    for name in a.only or MODELS:
        repo, lic, why = MODELS[name]
        allow, size = plan(repo)
        total += size
        print(f"{name:16} {repo:42} {size / 2**30:5.2f} GB  {lic:10} {why}")
        if not a.dry_run:
            path = snapshot_download(repo, allow_patterns=allow)
            print(f"{'':16} -> {path}")
    print(f"total: {total / 2**30:.2f} GB {'(dry run)' if a.dry_run else 'cached'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
