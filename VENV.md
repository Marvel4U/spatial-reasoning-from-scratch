# Environment

One Python 3.12 environment serves the whole repository: the data pipeline, the from-scratch model and harness, and the Qwen track. All runs in this repository used it, with torch 2.11.0+cu128; `requirements.txt` pins torch 2.14.0 (CUDA 13.0 wheels), with which the release check of `REPRODUCE.md` section 8 gives identical results.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # pipeline, model, harness, tests
pip install -r requirements-track-a.txt    # only for the Qwen track (qwen_reference/, qwen_fine_tuning/)
```

Without a system `python3.12`, `uv venv --seed --python 3.12 .venv` creates the same environment (`--seed` installs pip into it).

`requirements-track-a.txt` does not pin torch; it expects the torch from `requirements.txt` (2.14 with CUDA) to be installed already.

Confirm that the active interpreter is the one you installed into:

```bash
echo "$VIRTUAL_ENV"
python -c "import sys, torch; print(sys.executable, torch.__version__, torch.cuda.is_available())"
```
