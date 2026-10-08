"""Download Qwen/Qwen3-4B-Instruct-2507 to $HF_HOME on the HPC login node.

IITD compute nodes cannot reach huggingface.co (proxy-blocked). The login node
can, via the same basic-auth proxy used elsewhere. Run this ONCE after SFTP-ing
the code, before submitting training jobs.

Usage (on HPC login node):
    source $HOME/scratch/col775_a2_hpc/hpc/setup/proxy_env.sh  # or inline exports
    source $HOME/condaBaseSetup && conda activate col775_t25
    python $HOME/scratch/col775_a2_hpc/hpc/setup/prefetch_qwen.py
"""
from __future__ import annotations
import os, sys

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def main():
    hf_home = os.environ.get("HF_HOME") or os.path.join(os.environ["HOME"], "scratch", "col775_a2_hpc", "archive", "hf_cache")
    os.environ["HF_HOME"] = hf_home
    os.makedirs(hf_home, exist_ok=True)
    print(f"[prefetch] HF_HOME = {hf_home}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[prefetch] huggingface_hub missing; install via pip first.", file=sys.stderr)
        sys.exit(1)

    print(f"[prefetch] Downloading {MODEL_ID} snapshot …")
    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=None,           # use cache layout (HF_HOME/hub/models--…)
        cache_dir=hf_home,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt", "*.model"],
    )
    print(f"[prefetch] snapshot at: {path}")

    # Verify we can load it purely locally.
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True, cache_dir=hf_home)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True, cache_dir=hf_home)
    print(f"[prefetch] config.model_type = {cfg.model_type}  hidden_size = {cfg.hidden_size}")
    print(f"[prefetch] tokenizer vocab_size = {tok.vocab_size}")
    print("[prefetch] done.")


if __name__ == "__main__":
    main()
