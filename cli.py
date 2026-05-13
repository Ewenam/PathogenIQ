#!/usr/bin/env python3
"""
PathogenIQ CLI — Biosurveillance Intelligence Platform
"""
import click
from pathlib import Path


@click.group()
@click.version_option("0.1.0", prog_name="pathogeniq")
def cli():
    """PathogenIQ: AI-powered pathogen surveillance from sequencing data."""


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--config", "-c", default=None, help="Path to YAML config file.")
@click.option("--rank", "-r", default="G", show_default=True,
              type=click.Choice(["G", "S", "F"]),
              help="Taxonomic rank: G=genus, S=species, F=family.")
@click.option("--output", "-o", default="./reports", show_default=True,
              help="Output directory for reports.")
@click.option("--alert-threshold", default=0.6, show_default=True,
              help="Risk score [0–1] above which an alert is raised.")
@click.option("--characterize", is_flag=True, default=False,
              help="Run ESMFold structure prediction on flagged novel pathogens.")
@click.option("--quiet", is_flag=True, default=False, help="Suppress progress output.")
def run(input_path, config, rank, output, alert_threshold, characterize, quiet):
    """
    Run the full PathogenIQ pipeline on Kraken2 reports or a count matrix.

    INPUT_PATH: directory of *.report files, or path to a taxa×samples TSV/CSV.

    Examples:\n
      pathogeniq run ./kraken_reports/\n
      pathogeniq run counts.tsv --rank S --output ./out --characterize\n
      pathogeniq run ./reports/ --config configs/default.yaml
    """
    from pathogeniq.pipeline.runner import run as _run, PipelineConfig

    if config:
        cfg = PipelineConfig.from_yaml(config)
        cfg.output_dir = output
        cfg.alert_threshold = alert_threshold
    else:
        cfg = PipelineConfig(output_dir=output, alert_threshold=alert_threshold)

    results = _run(
        input_path=input_path,
        config=cfg,
        rank=rank,
        run_characterization=characterize,
        quiet=quiet,
    )

    alerts = results["alerts"]
    if alerts:
        click.echo(f"\n⚠  {len(alerts)} alert(s) detected.")
        for r in alerts:
            click.echo(f"   {r.level}: {r.sample_name} (score={r.score:.3f})")
    click.echo(f"\nReports saved to: {output}/")


@cli.command()
@click.argument("taxon_name")
@click.option("--no-fold", is_flag=True, default=False,
              help="Skip ESMFold structure prediction (NCBI lookup only).")
def characterize(taxon_name, no_fold):
    """
    Characterize a specific taxon: fetch protein sequences and predict structures.

    Example:\n
      pathogeniq characterize "Yersinia pestis"\n
      pathogeniq characterize Salmonella --no-fold
    """
    from pathogeniq.characterization.alphafold import characterize_novel_pathogen
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(f"\n[bold]Characterizing:[/bold] {taxon_name}")
    console.print("Fetching protein sequences from NCBI...")

    preds = characterize_novel_pathogen(taxon_name, use_esmfold=not no_fold)

    if not preds:
        console.print("[yellow]No protein sequences found for this taxon.[/yellow]")
        return

    table = Table(title=f"Characterization: {taxon_name}", show_lines=True)
    table.add_column("Protein", style="cyan", max_width=40)
    table.add_column("pLDDT", justify="center")
    table.add_column("Virulence Flags")
    table.add_column("Risk Assessment", max_width=50)

    for p in preds:
        table.add_row(
            p.protein_name[:40],
            f"{p.confidence:.1f}" if p.confidence > 0 else "—",
            ", ".join(p.virulence_flags) or "none",
            p.risk_annotation[:100],
        )

    console.print(table)


@cli.command()
@click.argument("report_json", type=click.Path(exists=True))
def summary(report_json):
    """Print a summary of an existing JSON report."""
    import json
    from rich.console import Console
    from rich.table import Table

    console = Console()
    with open(report_json) as f:
        data = json.load(f)

    s = data["summary"]
    console.print(f"\n[bold]PathogenIQ Report Summary[/bold]  ({data['generated_at']})")
    console.print(f"  Samples: {s['total_samples']}  |  "
                  f"[red]CRITICAL: {s['critical']}[/red]  "
                  f"[orange1]HIGH: {s['high']}[/orange1]  "
                  f"[yellow]MODERATE: {s['moderate']}[/yellow]  "
                  f"[green]LOW: {s['low']}[/green]")

    table = Table(show_header=True)
    table.add_column("Sample")
    table.add_column("Level")
    table.add_column("Score", justify="right")
    table.add_column("Top Pathogen")

    for sample in data["samples"]:
        color = {"LOW": "green", "MODERATE": "yellow", "HIGH": "red", "CRITICAL": "bold red"}.get(sample["level"], "white")
        top = sample["detected_pathogens"][0]["taxon"] if sample["detected_pathogens"] else "—"
        table.add_row(
            sample["name"],
            f"[{color}]{sample['level']}[/{color}]",
            str(sample["score"]),
            top,
        )
    console.print(table)


@cli.command()
@click.option("--db", default=None, envvar="PATHOGENIQ_DB",
              help="Path to history SQLite DB (default: ~/.pathogeniq/history.db).")
@click.option("--report", default="./reports/report.json", show_default=True,
              envvar="PATHOGENIQ_REPORT", help="Path to latest report.json.")
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host.")
@click.option("--port", default=8765, show_default=True, help="Bind port.")
def dashboard(db, report, host, port):
    """
    Launch the PathogenIQ real-time web dashboard.

    Reads data from the SQLite history store and the latest report.json.

    Example:\n
      pathogeniq dashboard\n
      pathogeniq dashboard --report ./reports/report.json --port 8765
    """
    import os
    import uvicorn

    if db:
        os.environ["PATHOGENIQ_DB"] = db
    os.environ["PATHOGENIQ_REPORT"] = report

    click.echo(f"PathogenIQ Dashboard → http://localhost:{port}")
    click.echo(f"  Report: {report}")
    uvicorn.run(
        "pathogeniq.dashboard.app:app",
        host=host,
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    cli()
