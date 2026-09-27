# OPEN_DECISIONS — not yet decided; do NOT silently decide these
Status 11 Sep 2026. Each needs Marvin's call (Claude proposes, Marvin decides — he defends every choice in interviews). Move decided items to the bottom with date + rationale.

## Open
0. **(m) Layers in v1** (proposal: buildings+heights, sidewalks, static GTFS, noise, computed shadow, POIs, trees). **(n) Which public benchmarks we commit to** (MapEval-API / -Visual / MapReason-OSM). See survey §5.
1. **(a) Tool API v1.** Minimal set at start — `zoom(bbox)`, `measure(a,b)`, `query_layer(pos, layer)`? Does `count_in_region` exist as a tool, or does counting stay a model skill (Phase 1.5 showed counting as a distinct failure mode — giving it away as a tool removes a measurable skill)? Also: tool-call syntax/format (JSON vs code-style), max calls per episode, error behavior on malformed calls.
2. **(b) Stage-1 trace source.** Scripted planner vs frontier distillation (rejection-sampled) vs mixed (leaning: scripted P0–P1, distilled P2). Cost estimate needed for the distillation API budget.
3. **(c) Model class.** 3–4B (cheaper GRPO, more dramatic deltas vs the 7% baseline) vs 7–8B (stronger starting perception). Also which family/version — check what's current at kickoff (Qwen-VL default assumption).
4. **(d) P3 scope.** In the repo as aspirational chapter, or out entirely until Stages 1–2 ship? (Lean: aspirational chapter.)
5. **(e) OSM transfer demo.** Commit to it (strong closer, extra work: real-layer plumbing) or synthetic-only for v1?
6. **(f) Authorship split for this project.** The sprint's learning rule was "Marvin types all core code." For this bigger build: which components are Marvin-typed (proposal: model/LoRA configs, GRPO loop, reward/verifier logic) vs collaborative (proposal: generator plumbing, eval infra, data pipelines, viewers)? Confirm explicitly.
7. **(g) Where DPO lives in the portfolio.** This project doesn't naturally exercise DPO. Bolt a small DPO chapter onto something cheap elsewhere, or let GRPO supersede it narratively? (Interviews test whether he can whiteboard both losses — that's independent of the repo.)
8. **(h) Repo name + public timing.** Working folder name `spatial_reasoning_LLM_artifact`; public GitHub name TBD. Publish early-and-iterate vs publish-when-Stage-1-done?
9. **(i) Relationship to the GPT-2 repro track.** It's still pending in `llm_qualification_projects` (FineWeb prep was going to be Marvin-with-Cursor). Sequencing: GPT-2 first, parallel, or after Stage 1?

## Decided
- **Two tracks on the same data/tasks/verifier (19 Sep):** Track A = fine-tune a Qwen-class VLM (portfolio credential, priority); Track B = "local", Marvin's own from-scratch vision model with native N-channel input, command tokens, optional 64x64 pointing output (control + measurement instrument). Details: `PLAN.md`.
- **P0 redefined (19 Sep):** the world as aligned, very simple raster layers (3-5 widely spaced classes each); first skill = relating the same place across layers. For Track A three layers are packed into RGB with a legend; later the model chooses which layers to view.
- **No further derived layers for now (19 Sep):** the 7 raw layers are enough to start; shadow / sidewalk width parked.
- **First task = T0 "point read"** (surface or noise class at a given point), then cross-layer point, area/mask, paths. Baseline the untrained model before any training. (19 Sep.)
- **Authorship (f) settled (19 Sep):** local model and all model/training code typed by Marvin, the local model only in sessions he leads line by line; data/task generators and eval infra collaborative with review.
- **(j) World substrate = real-city frozen snapshots**, not procedural worlds. Tools and verifiers read the same local datastore; "synthetic" = rendered images + generated tasks. (11 Sep, after survey: `survey/SURVEY_2026-09-11.md`.)
- **(k) Cities = Amsterdam first, then Barcelona, plus a Paris district and Vienna.** London and New York dropped (OS licensing; no noise map / no open licence). (11 Sep.)
- **(l) Licensing posture: deferred** until there are good results; note ODbL share-alike risk on a released QA corpus. (11 Sep.)
- **Opening hours deferred** as a task layer; stick to layers that are easy in the task stack (heights/shadow, noise, transit, sidewalks, POIs). (11 Sep.)
- **Paper mode:** no cover-to-cover reading; Marvin skims, we discuss implications together. Reading list: `survey/KEY_PAPERS.md`. (11 Sep.)
- **PERIA (Jun 2026) is a reproduction/branch-off candidate**; investigation in progress (`survey/raw/05_...`). (11 Sep.)
- **Topic = tool-using spatial agent on synthetic worlds** (11 Sep; see PROJECT_BRIEF for the full rejection list and rationale).
- **Tools enter early (P0), not as a final stage** (11 Sep, Marvin's reframing: "solve this real-world task, here is a map to get you started").
- **Ladder starts grounded-perceptual** despite early tools (11 Sep; Phase 1.5: perception is the bottleneck; tools need grounding to aim them).
- **Mini-R1 text exercise as GRPO pipeline debug** before the VLM run (11 Sep).
- **Frontier comparison framing:** identical tool API for everyone; frontier = reference ceiling, headline = base-vs-trained small model (11 Sep; "our model beats GPT-4o" framings rejected as high-risk unless conditions are exact).
