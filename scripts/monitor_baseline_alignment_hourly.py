"""Hourly liveness check for the active baseline-alignment queue."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time


TERMINAL_STATES = {"completed_artifact_acceptance_passed", "failed_stopped"}
FATAL_MARKERS = ("Traceback (most recent call last)", "CUDA out of memory", "Killed")


def alive(pid):
    if not pid:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().split()
        os.kill(int(pid), 0)
        return len(stat) > 2 and stat[2] != "Z"
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return False


def check(report):
    root = json.loads((report / "runs.json").read_text())
    anomalies = []
    workers = []
    if root.get("state") not in TERMINAL_STATES and not alive(root.get("coordinator_pid")):
        anomalies.append(f"coordinator pid {root.get('coordinator_pid')} is not alive")

    for model, record_path in root.get("models", {}).items():
        record = json.loads(Path(record_path).read_text())
        for run in record.get("runs", []):
            if run.get("state") != "training":
                continue
            pid_alive = alive(run.get("pid"))
            output_complete = (Path(run["output_dir"]) / "metrics.json").exists()
            log_path = Path(run["log"])
            tail = log_path.read_text(errors="ignore")[-20000:] if log_path.exists() else ""
            fatal = next((marker for marker in FATAL_MARKERS if marker in tail), None)
            workers.append({
                "model": model,
                "protocol": run["protocol"],
                "pid": run.get("pid"),
                "alive": pid_alive,
                "output_complete": output_complete,
            })
            if fatal:
                anomalies.append(f"{model}/{run['protocol']} log contains {fatal!r}")
            if not pid_alive and not output_complete:
                anomalies.append(
                    f"{model}/{run['protocol']} pid {run.get('pid')} exited without metrics.json"
                )

    entry = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "root_state": root.get("state"),
        "coordinator_pid": root.get("coordinator_pid"),
        "coordinator_alive": alive(root.get("coordinator_pid")),
        "active_workers": workers,
        "status": "anomaly" if anomalies else "healthy",
        "anomalies": anomalies,
    }
    with (report / "hourly_health_checks.jsonl").open("a") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False), flush=True)
    return root.get("state") in TERMINAL_STATES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    report = args.report.resolve()
    while True:
        terminal = check(report)
        if args.once or terminal:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
