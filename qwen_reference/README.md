# Qwen reference (Track A)

Vanilla **Qwen3-VL-2B** on `data/track_a/` exports. Run from repo root, project venv (`VENV.md`).

Artifacts default to **`runs/qwen_vanilla/`** (logs, jsonl, overlay PNGs).

| Module | Purpose |
|--------|---------|
| `coords.py` | 0–1000 ↔ image pixels; JSON parse |
| `load_model.py` | HF load + `generate` |
| `prompts.py` | v1 / v2 / marker + anchored route |
| `infer.py` | `v1`, `v2`, **`two-pass`** (markers then path) |
| `smoke.py` | describe, marker probe |
| `vanilla.py` | route inference + score |
| `overlay.py` | magenta=pred, blue=teacher PNG |
| `chat.py` | interactive stdin REPL (sanity-check the model) |
| `gpu.py` | `nvidia-smi`, torch `empty-cache` |

```bash
python -m qwen_reference.chat --image data/track_a/de_pijp_val_plain/images/route_000000.png --device-map cuda:0

python -m qwen_reference.smoke markers --route 0

# default: prints full chat prompt + raw reply before/after each generate; use --quiet to hide
python -m qwen_reference.vanilla one --route 0 --overlay

python -m qwen_reference.vanilla one --route 0 --prompt v2 --overlay
python -m qwen_reference.vanilla run --limit 20 --score

# A/B ~200 val routes: two-pass vs oracle GT endpoints (long run; use --quiet)
python -m qwen_reference.eval_ab --limit 200 --quiet --device-map cuda:0

# 4B same A/B (separate outputs: pred_4b_* , eval_ab_4b_n200.json) — see runs/qwen_vanilla/RUN_4B_VANILLA_AB.md
python -m qwen_reference.eval_ab --model models/hf/Qwen3-VL-4B-Instruct --tag 4b --limit 200 --quiet --device-map cuda:0
python -m qwen_reference.vanilla one --route 363 --prompt oracle --overlay
```

**Prompt modes**

- `v1` — single shot, list of points (original).
- `v2` — single shot, `{"start","goal","path"}` + explicit 0–1000 frame.
- `two-pass` — marker probe (works well), then route with **fixed** start/goal in the prompt.

Model: `models/hf/Qwen3-VL-2B-Instruct`. Data: `data/track_a/de_pijp_val_plain`.

**VRAM on the 3060:** check `python -m qwen_reference.gpu status`. If **ollama** holds ~9 GB,
stop it before Qwen runs or use `--device-map cuda:0` only after freeing GPU memory.
`empty-cache` clears torch’s cache in the current Python process only.
