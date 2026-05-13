"""
detection/kraken.py
Kraken2 subprocess wrapper — runs classification and parses results.
Falls back to loading pre-existing reports if Kraken2 is not installed.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..ingestion.reader import SampleSet, load_kraken_report, Sample


@dataclass
class Kraken2Config:
    db_path: str
    confidence: float = 0.1
    threads: int = 4
    min_hit_groups: int = 3


def kraken2_available() -> bool:
    return shutil.which("kraken2") is not None


def run_kraken2(
    fastq_r1: Path,
    output_dir: Path,
    config: Kraken2Config,
    fastq_r2: Path | None = None,
    sample_name: str | None = None,
) -> Path:
    """
    Run Kraken2 on a FASTQ file (or paired FASTQ). Returns path to the report file.
    Raises RuntimeError if Kraken2 is not installed.
    """
    if not kraken2_available():
        raise RuntimeError(
            "Kraken2 is not installed. Install it with:\n"
            "  conda install -c bioconda kraken2\n"
            "Or provide pre-computed .report files."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = sample_name or fastq_r1.stem
    report_path = output_dir / f"{name}.report"
    output_path = output_dir / f"{name}.kraken"

    cmd = [
        "kraken2",
        "--db", str(config.db_path),
        "--threads", str(config.threads),
        "--confidence", str(config.confidence),
        "--minimum-hit-groups", str(config.min_hit_groups),
        "--report", str(report_path),
        "--output", str(output_path),
    ]

    if fastq_r2 is not None:
        cmd += ["--paired", str(fastq_r1), str(fastq_r2)]
    else:
        cmd.append(str(fastq_r1))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Kraken2 failed:\n{result.stderr}")

    return report_path


def classify_samples(
    samples: list[Sample],
    output_dir: Path,
    config: Kraken2Config,
    rank: str = "G",
) -> SampleSet:
    """
    Run Kraken2 on each sample and return a SampleSet with taxa counts.
    Skips samples that already have a kraken_report set.
    """
    import pandas as pd

    series_list = []
    classified = []

    for sample in samples:
        if sample.kraken_report and sample.kraken_report.exists():
            report = sample.kraken_report
        elif sample.fastq_r1:
            report = run_kraken2(
                fastq_r1=sample.fastq_r1,
                output_dir=output_dir,
                config=config,
                fastq_r2=sample.fastq_r2,
                sample_name=sample.name,
            )
            sample = Sample(
                name=sample.name,
                kraken_report=report,
                fastq_r1=sample.fastq_r1,
                fastq_r2=sample.fastq_r2,
                metadata=sample.metadata,
            )
        else:
            raise ValueError(f"Sample '{sample.name}' has no FASTQ or existing report.")

        classified.append(sample)
        series_list.append(load_kraken_report(report, rank=rank))

    counts = pd.concat(series_list, axis=1).fillna(0)
    rel = counts.div(counts.sum(axis=0), axis=1).fillna(0)

    from ..ingestion.reader import SampleSet
    return SampleSet(samples=classified, taxa_matrix=counts, relative_abundance=rel)
