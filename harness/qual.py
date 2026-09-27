"""Post-train qualitative check (spatial metrics on val)."""
import config
import eval_spatial


def run_qual_sample(model, session: bool = False) -> dict:
    print("\n--- qual sample (spatial eval) ---")
    # c2 val loader is task-major: the first N batches are one task only — partial batches lie.
    max_batches = None if config.task_rung in ("c2", "c3", "c4") else 5
    result = eval_spatial.eval_model(model, max_batches=max_batches)
    eval_spatial.print_report(result)
    return result
