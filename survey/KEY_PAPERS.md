# Key papers — reading list (11 Sep 2026)
PDFs in `papers/<arxiv-id>.pdf`. Mode: Marvin skims, we discuss implications together; nobody reads cover to cover. Order = suggested discussion order.

## Tier 1 — closest to our project (discuss first)
| # | Paper | Date | arXiv | Why |
|---|---|---|---|---|
| 1 | **PERIA: Perceive, Interact, Reason** — tool-augmented spatial agent, PERIA-8B (Qwen3-8B), map tasks included, SFT + OR-GIGPO | Jun 2026 | [2606.12830](https://arxiv.org/abs/2606.12830) | Closest on "small VLM + spatial tools + SFT→RL". Reproduction candidate. |
| 2 | **MapReason-OSM** — 12k rendered OSM panels, 10 US downtowns, graph-verifiable tasks, CC BY 4.0 | Jun 2026 | [2606.22597](https://arxiv.org/abs/2606.22597) · [code](https://github.com/Vi-Sri/mapreason-osm) | Closest eval set; render pipeline reusable. |
| 3 | **MapAgent** — hierarchical planner + map-tool agent, SOTA on MapEval (prompted frontier LLMs) | Sep 2025 / EACL 2026 | [2509.05933](https://arxiv.org/abs/2509.05933) · [code](https://github.com/Hasebul/MapAgent) | Baseline scaffold we compete against. |
| 4 | **Spatial-Agent** — GeoFlow Graphs, MapEval-API | Jan 2026 | [2601.16965](https://arxiv.org/abs/2601.16965) | Second baseline on the same benchmark. |
| 5 | **MapEval** — the anchor benchmark (Textual/API/Visual) | Jan 2025 / ICML 2025 | [2501.00316](https://arxiv.org/abs/2501.00316) · [code](https://github.com/MapEval) | What we report on. |
| 6 | **Thinking with Map** (AMap) — Qwen3-VL-30B-A3B, 6 map tools incl. zoom, GRPO from base | Jan 2026 | [2601.05432](https://arxiv.org/abs/2601.05432) | Closest on the training axis (map tools + GRPO). |
| 7 | **OpenEarthAgent** — agentic SFT on 14.5k validated geo-tool trajectories, deterministic replay, fully open | Feb 2026 / ECCV 2026 | [2602.17665](https://arxiv.org/abs/2602.17665) · [code](https://github.com/mbzuai-oryx/OpenEarthAgent) | Trajectory-validation pipeline to copy. |

## Tier 2 — recipes and reward design
| # | Paper | Date | arXiv | Why |
|---|---|---|---|---|
| 8 | **SpatialLadder** — 3B, 26.6k samples, curriculum SFT→RLVR, beats GPT-4o | Oct 2025 / ICLR 2026 | [2510.08531](https://arxiv.org/abs/2510.08531) · [code](https://github.com/ZJU-REAL/SpatialLadder) | Curriculum template at our scale. |
| 9 | **SpatialThinker** — 7B, 7k synthetic samples, dense spatial rewards + online RL | Nov 2025 / NeurIPS 2025 workshop | [2511.07403](https://arxiv.org/abs/2511.07403) | Tiny-data RL precedent. |
| 10 | **ParaVT / PARA-GRPO** — RL for multiple tool calls in one turn, "tool prior paradox" | May 2026 | [2605.20342](https://arxiv.org/abs/2605.20342) | The many-tools-per-turn ambition. |
| 11 | **DeepEyes** — zoom-in tool RL, tool reward gated on actual use | May 2025 | [2505.14362](https://arxiv.org/abs/2505.14362) | Our `zoom` tool. |
| 12 | **Smooth Operator** — continuous numerical verifiable rewards | Jan 2026 | [2601.07695](https://arxiv.org/abs/2601.07695) | Distance/measure rewards. |
| 13 | **Active Exploring like a Pigeon** — programmatic spatial assertions as dense rewards | Jun 2026 / ICML 2026 | [2606.02459](https://arxiv.org/abs/2606.02459) | Reward design. |
| 14 | **MapTrace** — synthetic map supervision for route tracing | Dec 2025 | [2512.19609](https://arxiv.org/abs/2512.19609) | Synthetic-map SFT precedent. |

Full annotated list with ~50 more items: `raw/01_related_work_competition.md`.
