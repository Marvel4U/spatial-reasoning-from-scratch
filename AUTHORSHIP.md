# Authorship

This is Marvin Uhlmann's project. The research direction, the task ladder, every design decision and every training run are his. Two AI assistants worked in the loop, Claude (Anthropic) and Cursor Composer, and the split is recorded here because it matters for reading the results.

| component | who |
|---|---|
| overarching decisions, project plan, task ladder, what to test and when | Marvin |
| model (`plain_gpt_module/`: the grid vision transformer, relative position bias, conditioning) | Marvin |
| training harness, run ledger, diagnostics suite, experiment specs (`harness/`), and every training run | Marvin, with Cursor Composer support |
| data pipeline, task generators, GPU loaders, path-target builder, probes and analysis scripts (`worldsnap/`, `spatial_data/`, `analysis/`, `docs/plans/perf_prototypes/`) | Claude, reviewed by Marvin |
| reports, plans and this README | drafted by Claude on Marvin's questions, corrected and extended by Marvin |

Every number in the README names the run it comes from; the run ledger (`runs/experiments.json`) holds the configuration of each.
