# Drift reconciliation: source repository and this public copy (11 Oct 2026)

This copy of the repository was produced from a private source repository (the one the experiments ran in) and then
edited for publication. An external review found that of the 279 files tracked in both, 44 differed, and that some
of the differences were portability fixes in evaluation code that existed only here. This note classifies every one
of the 44, records what was carried back to the source repository, and states the rule that keeps the two from
drifting again.

Comparison: `git ls-files` in both repositories, contents compared with CRLF normalised to LF (line-ending-only
differences are not counted). Baseline: this branch at `abf7477`, source repository at `1500614`.

Classes:

- **a** publication scrub: machine names, home-directory paths, job-search wording; no change in meaning.
- **b** portability fix in code that the source repository also needs; carried back (see below).
- **c** public-only content: documents rewritten for publication, or files whose public form legitimately differs.
- **d** private-only content that must not go public (none among the 44; see the end of the note).
- **e** unclear: needs the author's decision.

## The 44 files

| File | Class | Reason |
|---|---|---|
| `.gitignore` | c | public copy tracks `runs/experiments.json` and the viewer panel notes; source-machine notes (data symlink to a second disk) dropped |
| `LEARNING_REPORT.md` | a | machine name replaced (since moved to `docs/archive/` by the concurrent housekeeping) |
| `OPEN_DECISIONS.md` | e | here decision (l) licensing is recorded as decided (MIT code/weights, ODbL 1.0 label stacks); the source repository still says deferred. A decision record, not a scrub: should be recorded in the source repository too |
| `PILOT_de_pijp_2026-09-12.md` | a | home-directory path replaced by "the repository root" (since moved to `docs/archive/`) |
| `VENV.md` | c | rewritten: create a fresh `.venv` from the requirements files instead of using the source machine's shared environment |
| `docs/memo/ITERATIVE_SEARCH_2026-09-25.md` | a | "superseded by C5_PLAN" sentence dropped; `docs/plans/C5_PLAN.md` is in this copy, so the pointer could stay |
| `docs/memo/SUNSHINE_LAYER_2026-09-23.md` | a | home-directory path |
| `docs/memo/TORCH_COMPILE_DYNAMIC_2026-10-01.md` | a | machine name |
| `docs/memo/TRACK_A_BACKBONES_2026-10-01.md` | a | environment path, cloud provider name |
| `docs/plans/ANALYSIS_SUITE_SPEC.md` | a | machine name |
| `docs/plans/C4_IMPLEMENTATION.md` | a | home-directory path |
| `docs/plans/PERFORMANCE_PLAN_2026-09-21.md` | a | machine name |
| `docs/plans/PERF_BASELINE_2026-09-21.md` | a | machine name |
| `docs/plans/TRACK_A_QWEN_PLAN_2026-10-04.md` | a | cloud provider and job-search wording; status line rewritten in the past tense with a pointer to the results report |
| `docs/plans/perf_prototypes/c1_logit_correction_check.py` | b | repository root from `Path(__file__)`, no home-directory fallback |
| `docs/plans/perf_prototypes/c1_target_stats.py` | b | same |
| `docs/plans/perf_prototypes/c2_target_stats.py` | b | same |
| `docs/plans/perf_prototypes/c3_head_sink_probe.py` | b | same |
| `docs/plans/perf_prototypes/c3_m1_shortcut_probe.py` | b | same |
| `docs/plans/perf_prototypes/c3_marker_count_probe.py` | b | same |
| `docs/plans/perf_prototypes/c3_outside_disc_probe.py` | b | same |
| `docs/plans/perf_prototypes/c3_radius_stats.py` | b | same |
| `docs/plans/perf_prototypes/c3_relative_stats.py` | b | same |
| `docs/plans/perf_prototypes/c3_signal_coverage.py` | b | same |
| `docs/plans/perf_prototypes/c3_vark_loader_probe.py` | b | same |
| `docs/plans/perf_prototypes/c3_wrong_token_probe.py` | b | same |
| `docs/plans/perf_prototypes/c4_pertask_eval.py` | b | same |
| `docs/plans/perf_prototypes/c4_tolerant_metrics.py` | b | same |
| `docs/plans/perf_prototypes/c5_length_check.py` | b | same |
| `docs/plans/perf_prototypes/c5_rays_loader_check.py` | b | same |
| `docs/reports/C4_SEGMENT_DETOUR_2026-09-25.md` | a | `cd <home path> &&` dropped from a command line |
| `docs/reports/C5_WAVE_SEARCH_2026-10-03.md` | c | "draft for review" and pending-marker sentence dropped (no pending markers remain); figures point to `docs/figures/` instead of the source repository's memo folders; README draft pointer now points to `README.md` (since 11 Oct the paper text is `PAPER.md`) |
| `docs/reports/TRACK_A_QWEN_VANILLA_AND_SFT_2026-10-08.md` | a | job-search and "product decision" wording, environment path |
| `harness/diagnostics/c4_reference_pred_strip.py` | b | `_apply_meta_config` no longer forces the training machine's directories from `meta.json` |
| `harness/diagnostics/drawing_to_polyline.py` | b | repo-relative `eval_json` in a released pads file resolved against the repository root |
| `harness/diagnostics/wave_search_rollout.py` | b | `_apply_meta_config` skips machine directory keys (`MACHINE_PATH_KEYS`) from `meta.json` |
| `harness/experiments.py` | a | two docstrings: job-search wording replaced by "reference" |
| `harness/run_overnight_30sep.sh` | a | home-directory path in a usage comment |
| `qwen_reference/README.md` | a | "sprint venv" wording |
| `qwen_reference/gpu.py` | a | machine name in the module docstring |
| `requirements-track-a.txt` | c | install instructions for this repository's own `.venv` |
| `requirements.txt` | c | flat pinned list for a fresh environment (adds pillow, pytest, torch) instead of the annotated source-machine file |
| `tests/test_c0_gpu_loader.py` | b | skip when the c0 task data is not built |
| `tests/test_c2_tasks.py` | b | skip when the c2 task statistics are not built |

Totals: a 17, b 21, c 5, d 0, e 1.

## Carried back to the source repository

All 21 class-b changes are in the source repository's `main` as commit **`e1c6c04`** ("Portability back-port from the
public copy: meta.json machine paths, relative pads paths, test skips, script roots"); its full test suite passed
(167 passed). Twenty of the files are now identical to this copy (CRLF normalised). One difference remains on
purpose, in `_apply_meta_config` of `wave_search_rollout.py` and `c4_reference_pred_strip.py`:

- this copy skips the machine directory keys unconditionally;
- the source repository skips them only when the recorded directory does not exist (or cannot be read) on the
  current machine, so runs on the training machine keep reading the directories recorded in their `meta.json`.

From any other clone the two behave the same. The next export from the source repository will replace this copy's
version with the source version.

## Other drift found (not among the 44)

- `docs/figures_make.py`, `docs/figures_make2.py` exist only in this copy (README figure scripts; they already use
  `Path(__file__)` roots). Class **e**: code that is not in the source repository, against the rule below. They read
  `runs/` and `data/` of a full working copy, so their natural home is the source repository.
- Housekeeping in progress during this reconciliation (commits `44c29bc` to `099310c`): root planning documents moved
  to `docs/archive/`, line endings normalised, licence and requirements files edited. It also touched
  comments and docstrings in code (`harness/experiments.py`, `plain_gpt_module/local_config.py`, and line endings in
  several `spatial_data/` and test files) to follow the new archive paths. These are comment-only, but they are code
  files edited here and not in the source repository; the next export would revert them unless the scrub re-applies
  them.
- Files that exist only in the source repository (assistant instructions, planning and job-search documents, the
  predecessor project's harness, third-party model reference code, draft reports) are correctly absent here and stay
  out (class d by file, not by diff).

## Rule from here on

1. **Code flows one way: source repository to public copy.** The public copy is refreshed with a `git archive` of
   the source repository's tracked files, followed by the publication scrub (machine names, home-directory paths,
   job-search wording, private-only files removed).
2. **Documents may diverge.** Reports, plans and root documents may be rewritten for publication here; that is
   class a and c drift and is expected.
3. **The public copy never gets a code edit that is not first in the source repository.** A fix found while
   preparing the publication is made in the source repository, committed there, and arrives here with the next
   export. Comment-only path updates that follow a public-only document layout belong in the scrub step, not in
   hand edits.
4. Decisions (such as `OPEN_DECISIONS.md` entries) are recorded in the source repository first, like code.
