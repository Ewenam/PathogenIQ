"""
scheduler/watcher.py
Directory watcher — polls for new data and runs the pipeline automatically.

Watches a directory for new Kraken2 report subdirectories or count matrix files.
On detection, runs the full PathogenIQ pipeline and saves reports.
Processed paths are tracked in .pathogeniq_processed.json to avoid re-running.

Usage:
    pathogeniq watch ./data --output ./reports --interval 3600
    pathogeniq watch ./data --interval 1800 --alert-email --alert-slack
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

_PROCESSED_FILE = ".pathogeniq_processed.json"


def _load_processed(data_dir: Path) -> set[str]:
    record = data_dir / _PROCESSED_FILE
    if record.exists():
        try:
            return set(json.loads(record.read_text()))
        except Exception:
            pass
    return set()


def _save_processed(data_dir: Path, processed: set[str]) -> None:
    record = data_dir / _PROCESSED_FILE
    record.write_text(json.dumps(sorted(processed), indent=2))


def _discover_new_inputs(data_dir: Path, processed: set[str]) -> list[Path]:
    """
    Find new, unprocessed inputs in data_dir:
      - Subdirectories containing *.report files → treated as Kraken2 report dirs
      - *.tsv / *.csv files → treated as count matrices
    """
    candidates: list[Path] = []

    # Count matrix files in root of data_dir
    for ext in ("*.tsv", "*.csv", "*.txt"):
        for f in sorted(data_dir.glob(ext)):
            if str(f) not in processed:
                candidates.append(f)

    # Subdirectories with Kraken2 reports
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir() and str(subdir) not in processed:
            reports = list(subdir.glob("*.report")) + list(subdir.glob("*.txt"))
            if reports:
                candidates.append(subdir)

    return candidates


def _run_pipeline(
    input_path: Path,
    output_dir: Path,
    alert_email: bool = False,
    alert_slack: bool = False,
    quiet: bool = False,
) -> bool:
    """Run PathogenIQ pipeline on one input. Returns True on success."""
    from pathogeniq.pipeline.runner import run, PipelineConfig

    out = output_dir / f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        output_dir=str(out),
        alert_email=alert_email,
        alert_slack=alert_slack,
    )

    try:
        run(input_path=input_path, config=cfg, quiet=quiet)
        # Symlink/copy latest report.json so dashboard always reads latest
        latest = output_dir / "report.json"
        src = out / "report.json"
        if src.exists():
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(src.resolve())
        return True
    except Exception as exc:
        print(f"  [watcher] Pipeline failed for {input_path}: {exc}")
        return False


def watch(
    data_dir: str | Path,
    output_dir: str | Path = "./reports",
    interval: int = 3600,
    alert_email: bool = False,
    alert_slack: bool = False,
    run_on_start: bool = False,
    quiet: bool = False,
) -> None:
    """
    Poll data_dir every `interval` seconds.
    Run the full pipeline whenever new data appears.

    Args:
        data_dir:     Directory to watch for new Kraken2 reports or count matrices
        output_dir:   Where to save pipeline reports (default ./reports)
        interval:     Poll interval in seconds (default 3600 = hourly)
        alert_email:  Fire email alert on HIGH/CRITICAL findings
        alert_slack:  Fire Slack alert on HIGH/CRITICAL findings
        run_on_start: Process existing unprocessed data immediately on launch
        quiet:        Suppress pipeline progress output
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"PathogenIQ Watcher started")
    print(f"  Watching:  {data_dir.resolve()}")
    print(f"  Output:    {output_dir.resolve()}")
    print(f"  Interval:  {interval}s ({interval/60:.0f} min)")
    print(f"  Alerts:    email={alert_email}  slack={alert_slack}")
    print(f"  Press Ctrl+C to stop.\n")

    processed = _load_processed(data_dir)
    if not run_on_start:
        # Mark existing data as already processed so only new data triggers runs
        existing = _discover_new_inputs(data_dir, set())
        for inp in existing:
            processed.add(str(inp))
        _save_processed(data_dir, processed)
        print(f"  Marked {len(existing)} existing input(s) as processed (use --run-on-start to process them).")

    try:
        while True:
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            new_inputs = _discover_new_inputs(data_dir, processed)

            if new_inputs:
                print(f"\n[{now}] Found {len(new_inputs)} new input(s):")
                for inp in new_inputs:
                    print(f"  → {inp}")
                    ok = _run_pipeline(inp, output_dir, alert_email, alert_slack, quiet)
                    if ok:
                        processed.add(str(inp))
                        _save_processed(data_dir, processed)
                        print(f"    ✓ Processed successfully")
                    else:
                        print(f"    ✗ Failed — will retry next cycle")
            else:
                print(f"[{now}] No new data. Next check in {interval}s ...")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nWatcher stopped.")
