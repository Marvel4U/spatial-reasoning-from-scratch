# Qwen fine-tuning (Track A routes)

Decisions (Oct 2026): **2B**, **single-shot** prompt at eval, train **100 → 1k → …**, drop **`leaves_window`** rows, teacher polyline in **0–1000** JSON.

Shared helpers: `qwen_reference/` (`coords`, `prompts`, `load_model`).

## SFT pipeline smoke test

```bash
python -m qwen_fine_tuning.train_batch_smoke --device-map cuda:0
python -m qwen_fine_tuning.train_batch_smoke --route 42 --device-map cuda:0
```

One train example (`route_v1` prompt + teacher polyline in 0–1000 JSON), masked CE forward.

## LoRA training (100 routes)

See `runs/qwen_sft/RUN_LORA100_V1.md`.

```bash
python -m qwen_fine_tuning.train_lora --device-map cuda:0
# long runs: checkpoint every 500 opt steps; after interrupt, same --output-dir + --resume
python -m qwen_fine_tuning.train_lora --output-dir runs/qwen_sft/lora20000_v1 --checkpoint-every 500 --resume --device-map cuda:0
```
