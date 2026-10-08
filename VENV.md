# Environment

One Python 3.12 environment serves the whole repository: the data pipeline, the from-scratch model and harness, and the Qwen track. All runs in this repository used it.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # pipeline, model, harness, tests
pip install -r requirements-track-a.txt    # only for the Qwen track (qwen_reference/, qwen_fine_tuning/)
```

`requirements-track-a.txt` does not pin torch; it expects the torch from `requirements.txt` (2.14 with CUDA) to be installed already.

Confirm that the active interpreter is the one you installed into:

```bash
echo "$VIRTUAL_ENV"
python -c "import sys, torch; print(sys.executable, torch.__version__, torch.cuda.is_available())"
```
