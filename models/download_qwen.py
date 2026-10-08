"""Track A backbones: the two smallest Qwen VL models (1 Oct 2026). Run once; weights are gitignored."""
from huggingface_hub import snapshot_download
for repo in ["Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3.5-0.8B"]:  # add Qwen/Qwen3-VL-4B-Instruct if needed
    p = snapshot_download(repo, local_dir="models/hf/" + repo.split("/")[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py", "*.md", "*.jinja"])
    print("done", repo, p, flush=True)
