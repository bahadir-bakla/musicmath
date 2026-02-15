import os
import subprocess
import time
from pathlib import Path
from celery_app import celery

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))


def _run_script(cmd: list[str], task, step_name: str, cwd: str | None = None) -> dict:
    work_dir = cwd or str(DATA_DIR)
    task.update_state(state="PROGRESS", meta={"step": step_name, "status": "running"})
    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    elapsed = round(time.time() - start, 1)
    return {
        "step": step_name,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:] if result.stdout else "",
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
        "elapsed_s": elapsed,
    }


@celery.task(bind=True, name="tasks.run_pipeline")
def run_pipeline(self):
    return _run_script(
        ["python", "run_pipeline.py"],
        self, "core_pipeline",
    )


@celery.task(bind=True, name="tasks.run_analyses")
def run_analyses(self):
    return _run_script(
        ["python", "run_pipeline.py", "--skip-pipeline", "--run-analyses"],
        self, "all_analyses",
    )


@celery.task(bind=True, name="tasks.run_brute_force_chunk")
def run_brute_force_chunk(self, chunk: int, total_chunks: int):
    return _run_script(
        ["python", "scripts/brute_force_motif.py",
         "--chunk", str(chunk), "--total-chunks", str(total_chunks)],
        self, f"brute_force_chunk_{chunk}",
    )


@celery.task(bind=True, name="tasks.run_brute_force_merge")
def run_brute_force_merge(self, total_chunks: int):
    return _run_script(
        ["python", "scripts/brute_force_motif.py",
         "--merge-chunks", "--total-chunks", str(total_chunks)],
        self, "brute_force_merge",
    )


@celery.task(bind=True, name="tasks.run_pattern_export")
def run_pattern_export(self):
    return _run_script(
        ["python", "scripts/brute_force_patterns.py", "--export-patterns"],
        self, "pattern_export",
    )


@celery.task(bind=True, name="tasks.run_brute_force_full")
def run_brute_force_full(self, total_chunks: int = 4):
    results = []

    self.update_state(state="PROGRESS", meta={
        "step": "brute_force_parallel",
        "status": "dispatching_chunks",
        "total_chunks": total_chunks,
    })
    chunk_tasks = []
    for i in range(total_chunks):
        t = run_brute_force_chunk.delay(i, total_chunks)
        chunk_tasks.append(t)

    for idx, t in enumerate(chunk_tasks):
        self.update_state(state="PROGRESS", meta={
            "step": "brute_force_parallel",
            "status": f"waiting_chunk_{idx}",
            "completed": idx,
            "total_chunks": total_chunks,
        })
        r = t.get(timeout=3600)
        results.append(r)

    self.update_state(state="PROGRESS", meta={
        "step": "brute_force_merge",
        "status": "merging",
    })
    merge_result = _run_script(
        ["python", "scripts/brute_force_motif.py",
         "--merge-chunks", "--total-chunks", str(total_chunks)],
        self, "brute_force_merge",
    )
    results.append(merge_result)

    return {"chunks": results, "status": "completed"}


@celery.task(bind=True, name="tasks.run_full_pipeline")
def run_full_pipeline(self, brute_force_chunks: int = 4):
    steps = []

    self.update_state(state="PROGRESS", meta={"step": "core_pipeline", "status": "running", "progress": "1/5"})
    r = _run_script(["python", "run_pipeline.py"], self, "core_pipeline")
    steps.append(r)
    if r["returncode"] != 0:
        return {"steps": steps, "status": "failed", "failed_at": "core_pipeline"}

    self.update_state(state="PROGRESS", meta={"step": "analyses", "status": "running", "progress": "2/5"})
    r = _run_script(
        ["python", "run_pipeline.py", "--skip-pipeline", "--run-analyses"],
        self, "analyses",
    )
    steps.append(r)
    if r["returncode"] != 0:
        return {"steps": steps, "status": "failed", "failed_at": "analyses"}

    self.update_state(state="PROGRESS", meta={"step": "brute_force", "status": "running", "progress": "3/5"})
    chunk_tasks = []
    for i in range(brute_force_chunks):
        t = run_brute_force_chunk.delay(i, brute_force_chunks)
        chunk_tasks.append(t)
    for t in chunk_tasks:
        r = t.get(timeout=3600)
        steps.append(r)
        if r["returncode"] != 0:
            return {"steps": steps, "status": "failed", "failed_at": f"brute_force_chunk"}

    self.update_state(state="PROGRESS", meta={"step": "brute_force_merge", "status": "running", "progress": "4/5"})
    r = _run_script(
        ["python", "scripts/brute_force_motif.py",
         "--merge-chunks", "--total-chunks", str(brute_force_chunks)],
        self, "brute_force_merge",
    )
    steps.append(r)
    if r["returncode"] != 0:
        return {"steps": steps, "status": "failed", "failed_at": "brute_force_merge"}

    self.update_state(state="PROGRESS", meta={"step": "pattern_export", "status": "running", "progress": "5/5"})
    r = _run_script(
        ["python", "scripts/brute_force_patterns.py", "--export-patterns"],
        self, "pattern_export",
    )
    steps.append(r)

    failed = [s for s in steps if s["returncode"] != 0]
    return {
        "steps": steps,
        "status": "failed" if failed else "completed",
        "failed_at": failed[0]["step"] if failed else None,
    }
