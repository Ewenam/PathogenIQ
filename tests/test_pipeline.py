"""End-to-end pipeline integration test using fixture reports."""
import pytest
from pathlib import Path


def test_full_pipeline_runs(two_sample_dir, tmp_path):
    """Pipeline should complete without error on two fixture samples."""
    from pathogeniq.pipeline.runner import run, PipelineConfig

    cfg = PipelineConfig(output_dir=str(tmp_path), alert_threshold=0.6)
    results = run(input_path=two_sample_dir, config=cfg, rank="G",
                  run_characterization=False, quiet=True)

    assert "risk_scores" in results
    assert "alerts" in results
    assert len(results["risk_scores"]) == 2

    # Report JSON must be written
    assert (tmp_path / "report.json").exists()


def test_pipeline_report_json_schema(two_sample_dir, tmp_path):
    """report.json must contain required top-level keys."""
    import json
    from pathogeniq.pipeline.runner import run, PipelineConfig

    cfg = PipelineConfig(output_dir=str(tmp_path))
    run(input_path=two_sample_dir, config=cfg, rank="G",
        run_characterization=False, quiet=True)

    report = json.loads((tmp_path / "report.json").read_text())
    for key in ("generated_at", "summary", "samples"):
        assert key in report, f"report.json missing key: {key}"

    for sample in report["samples"]:
        for field in ("name", "score", "level", "detected_pathogens", "amr_annotations"):
            assert field in sample, f"sample missing field: {field}"


def test_sample_b_alerts(two_sample_dir, tmp_path):
    """sample_b (Yersinia + Vibrio + Salmonella) should trigger at least one alert."""
    from pathogeniq.pipeline.runner import run, PipelineConfig

    cfg = PipelineConfig(output_dir=str(tmp_path), alert_threshold=0.5)
    results = run(input_path=two_sample_dir, config=cfg, rank="G",
                  run_characterization=False, quiet=True)
    alert_names = {r.sample_name for r in results["alerts"]}
    assert "sample_b" in alert_names


def test_watcher_discovers_new_input(tmp_path):
    """Watcher should detect a new report directory and mark it as new."""
    from pathogeniq.scheduler.watcher import _discover_new_inputs, _load_processed
    import shutil
    fixtures = Path(__file__).parent / "fixtures"

    data_dir = tmp_path / "watch_data"
    data_dir.mkdir()

    # No inputs yet
    assert _discover_new_inputs(data_dir, set()) == []

    # Add a subdir with reports
    subdir = data_dir / "run_20251019"
    subdir.mkdir()
    shutil.copy(fixtures / "sample_a.report", subdir / "sample_a.report")

    found = _discover_new_inputs(data_dir, set())
    assert len(found) == 1
    assert found[0] == subdir

    # After marking processed, should not appear again
    found2 = _discover_new_inputs(data_dir, {str(subdir)})
    assert found2 == []
