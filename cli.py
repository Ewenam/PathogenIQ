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
@click.option("--format", "-f", "report_format", default="auto", show_default=True,
              type=click.Choice(["auto", "kraken2", "bracken", "metaphlan"]),
              help="Classifier report format. 'auto' content-sniffs each file.")
@click.option("--output", "-o", default="./reports", show_default=True,
              help="Output directory for reports.")
@click.option("--alert-threshold", default=0.6, show_default=True,
              help="Risk score [0–1] above which an alert is raised.")
@click.option("--characterize", is_flag=True, default=False,
              help="Run ESMFold structure prediction on flagged novel pathogens.")
@click.option("--use-vqvae", is_flag=True, default=False,
              help="Enable VQ-VAE sequence-level novelty detection (requires trained model).")
@click.option("--rank-bump", is_flag=True, default=False,
              help="Bump low-prevalence/low-read species up to genus instead of dropping them.")
@click.option("--alert-email", is_flag=True, default=False,
              help="Send email alerts on HIGH/CRITICAL findings (requires PATHOGENIQ_SMTP_* env vars).")
@click.option("--alert-slack", is_flag=True, default=False,
              help="Send Slack alerts on HIGH/CRITICAL findings (requires PATHOGENIQ_SLACK_WEBHOOK).")
@click.option("--run-date", default=None,
              help="ISO date (YYYY-MM-DD) to stamp this run in the history store. "
                   "Defaults to today. Use this when replaying historical samples.")
@click.option("--external-signals", default=None, type=click.Path(exists=True),
              help="Path to a site/date/value CSV or TSV (e.g. qPCR, ddPCR, case counts) "
                   "to cross-validate temporal alerts against.")
@click.option("--quiet", is_flag=True, default=False, help="Suppress progress output.")
def run(input_path, config, rank, report_format, output, alert_threshold, characterize,
        use_vqvae, rank_bump, alert_email, alert_slack, run_date, external_signals, quiet):
    """
    Run the full PathogenIQ pipeline on classifier reports or a count matrix.

    INPUT_PATH: directory of Kraken2/Bracken/MetaPhlAn report files, or path to
    a taxa×samples TSV/CSV.

    Examples:\n
      pathogeniq run ./kraken_reports/\n
      pathogeniq run ./bracken_reports/ --format bracken\n
      pathogeniq run counts.tsv --rank S --output ./out --characterize\n
      pathogeniq run ./reports/ --config configs/default.yaml
    """
    import os
    from pathogeniq.pipeline.runner import run as _run, PipelineConfig

    actor = os.environ.get("PATHOGENIQ_ACTOR") or os.environ.get("USER") or "cli"

    if config:
        cfg = PipelineConfig.from_yaml(config)
        cfg.output_dir = output
        cfg.alert_threshold = alert_threshold
    else:
        cfg = PipelineConfig(output_dir=output, alert_threshold=alert_threshold)

    cfg.use_vqvae = use_vqvae
    cfg.rank_bump = rank_bump
    cfg.alert_email = alert_email
    cfg.alert_slack = alert_slack
    if run_date:
        cfg.run_date = run_date

    results = _run(
        input_path=input_path,
        config=cfg,
        rank=rank,
        format=report_format,
        run_characterization=characterize,
        quiet=quiet,
        actor=actor,
        external_signals=external_signals,
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
@click.option("--alert-threshold", default=0.6, show_default=True,
              help="Risk score threshold for alerting (0–1).")
@click.option("--quick", is_flag=True, default=False,
              help="Skip graph/SBM — fast validation of risk scoring only.")
@click.option("--seed", default=42, show_default=True, help="Random seed for synthetic data.")
@click.option("--output", "-o", default=None,
              help="Save benchmark results to this JSON file.")
@click.option("--no-mock-community", is_flag=True, default=False,
              help="Skip the ZymoBIOMICS mock-community validation scenario.")
def benchmark(alert_threshold, quick, seed, output, no_mock_community):
    """
    Run the full benchmark suite against synthetic ground-truth datasets.

    Generates 6 controlled scenarios (negative controls, low/high contamination,
    critical pathogens, multi-pathogen communities, novel agents) plus a
    ZymoBIOMICS reference-community scenario, and computes sensitivity,
    specificity, precision, and F1 for the detection pipeline.

    Example:\n
      pathogeniq benchmark\n
      pathogeniq benchmark --quick --output results.json
    """
    from pathogeniq.benchmark.runner import run_benchmark

    results = run_benchmark(
        alert_threshold=alert_threshold,
        quick=quick,
        seed=seed,
        include_mock_community=not no_mock_community,
    )

    if output:
        import json
        from pathlib import Path
        Path(output).write_text(json.dumps(results, indent=2))
        click.echo(f"\nResults saved → {output}")


@cli.command()
@click.argument("report_path", type=click.Path(exists=True))
@click.option("--threshold", default=0.001, show_default=True,
              help="Minimum relative abundance to report (0–1).")
def lineage(report_path, threshold):
    """
    Annotate species-level lineage context from a single Kraken2 report.

    Parses the report at species rank and looks up known outbreak strains,
    variants, and WHO priority context for any detected species.

    Example:\n
      pathogeniq lineage sample.report\n
      pathogeniq lineage sample.report --threshold 0.005
    """
    from pathogeniq.lineage.detector import load_species_from_report
    from rich.console import Console
    from rich.table import Table

    console = Console()
    annotations = load_species_from_report(report_path, threshold=threshold)

    if not annotations:
        console.print("[dim]No annotated species detected above threshold.[/dim]")
        return

    table = Table(title=f"Lineage Context: {report_path}", show_lines=True)
    table.add_column("Species", style="cyan", max_width=45)
    table.add_column("Abund%", justify="right")
    table.add_column("WHO", justify="center")
    table.add_column("Known Variants", max_width=40)
    table.add_column("WBE Note", max_width=45)

    for a in annotations:
        variants = "; ".join(v["name"] for v in a.known_variants[:2])
        table.add_row(
            f"{a.species}\n[dim]{a.common_name}[/dim]",
            f"{a.abundance*100:.2f}%",
            a.who_label,
            variants or "—",
            a.wbe_utility[:80] or "—",
        )
    console.print(table)


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--output", "-o", default="./reports", show_default=True,
              help="Directory for pipeline output reports.")
@click.option("--interval", default=3600, show_default=True,
              help="Poll interval in seconds.")
@click.option("--alert-email", is_flag=True, default=False,
              help="Send email alerts (requires PATHOGENIQ_SMTP_* env vars).")
@click.option("--alert-slack", is_flag=True, default=False,
              help="Send Slack alerts (requires PATHOGENIQ_SLACK_WEBHOOK env var).")
@click.option("--run-on-start", is_flag=True, default=False,
              help="Process existing unprocessed data immediately on launch.")
@click.option("--quiet", is_flag=True, default=False,
              help="Suppress pipeline progress output.")
def watch(data_dir, output, interval, alert_email, alert_slack, run_on_start, quiet):
    """
    Watch a directory and run the pipeline automatically on new data.

    Monitors DATA_DIR for new Kraken2 report subdirectories or count matrix
    files (*.tsv, *.csv). Runs the full PathogenIQ pipeline on each new input,
    saves reports to OUTPUT, and sends alerts if configured.

    Example:\n
      pathogeniq watch ./data\n
      pathogeniq watch ./data --interval 1800 --alert-slack --run-on-start
    """
    from pathogeniq.scheduler.watcher import watch as _watch
    _watch(
        data_dir=data_dir,
        output_dir=output,
        interval=interval,
        alert_email=alert_email,
        alert_slack=alert_slack,
        run_on_start=run_on_start,
        quiet=quiet,
    )


@cli.command("train-embedder")
@click.argument("fasta_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None,
              help="Checkpoint output path (default: ~/.pathogeniq/vqvae.pt).")
@click.option("--epochs", default=50, show_default=True, help="Training epochs.")
@click.option("--hidden-dim", default=256, show_default=True, help="Encoder hidden size.")
@click.option("--latent-dim", default=64, show_default=True, help="Latent (codebook) dimension.")
@click.option("--codebook-size", default=512, show_default=True, help="VQ codebook size.")
@click.option("--batch-size", default=32, show_default=True, help="Mini-batch size.")
@click.option("--lr", default=1e-3, show_default=True, help="Adam learning rate.")
@click.option("--device", default="cpu", show_default=True,
              type=click.Choice(["cpu", "cuda", "mps"]), help="Compute device.")
@click.option("--quiet", is_flag=True, default=False, help="Suppress progress output.")
def train_embedder(fasta_path, output, epochs, hidden_dim, latent_dim,
                   codebook_size, batch_size, lr, device, quiet):
    """
    Train a VQ-VAE on reference genomic sequences for sequence-level novelty detection.

    FASTA_PATH: FASTA file of reference genomes / assembled contigs.

    The trained model learns a discrete codebook of normal sequence profiles.
    At run time, sequences that diverge from the reference corpus yield high
    reconstruction error, boosting the novelty signal.

    Example:\n
      pathogeniq train-embedder refseq_bacteria.fasta --epochs 100\n
      pathogeniq train-embedder viral_refseq.fasta --device mps --codebook-size 1024
    """
    from pathogeniq.embedding.train import train, DEFAULT_OUTPUT
    from pathlib import Path

    out = Path(output) if output else DEFAULT_OUTPUT
    click.echo(f"\nTraining VQ-VAE sequence embedder")
    click.echo(f"  Input:    {fasta_path}")
    click.echo(f"  Output:   {out}")
    click.echo(f"  Epochs:   {epochs}  |  Codebook: {codebook_size}  |  Device: {device}\n")

    train(
        fasta_path=fasta_path,
        output_path=out,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_embeddings=codebook_size,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        quiet=quiet,
    )
    click.echo(f"\nCheckpoint saved → {out}")
    click.echo("Enable in pipeline: pathogeniq run ... --use-vqvae")


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


@cli.command()
@click.argument("report_dir", type=click.Path(exists=True, file_okay=False))
def verify(report_dir):
    """
    Verify a run's provenance manifest against the files on disk today.

    Re-hashes report.json's sample data and every recorded input file still
    present on disk, comparing against the values recorded in provenance.json
    at run time. Exits 1 if any check fails — scriptable for CI or
    regulator-facing automation.

    Example:\n
      pathogeniq verify ./reports/\n
      pathogeniq verify /tmp/piq_demo
    """
    from pathogeniq import provenance
    from rich.console import Console

    console = Console()
    result = provenance.verify_manifest(report_dir)

    for check in result["checks"]:
        if check["passed"]:
            console.print(f"  [green]PASS[/green]  {check['name']}  — {check['detail']}")
        else:
            console.print(f"  [bold red]FAIL[/bold red]  {check['name']}  — {check['detail']}")

    if result["all_passed"]:
        console.print(f"\n[bold green]All checks passed[/bold green] — {report_dir}")
    else:
        console.print(f"\n[bold red]Verification FAILED[/bold red] — {report_dir}")
        raise SystemExit(1)


@cli.command()
@click.option("--prevalence", required=True, type=float,
              help="Expected pathogen prevalence (fraction of reads), e.g. 0.0001.")
@click.option("--depth", default=0, show_default=True,
              help="Sequencing depth (total reads) to evaluate. Omit to only see required-depth targets.")
@click.option("--min-reads", default=1, show_default=True,
              help="Minimum reads required to call a detection.")
@click.option("--cost-per-million-reads", default=5.0, show_default=True,
              help="Sequencing cost per million reads ($).")
@click.option("--sample-prep-cost", default=50.0, show_default=True,
              help="Fixed per-sample prep cost ($).")
def plan(prevalence, depth, min_reads, cost_per_million_reads, sample_prep_cost):
    """
    Pre-deployment sequencing sensitivity/cost planning calculator.

    Estimates detection probability and cost for a given sequencing depth,
    and the depth required to hit 90/95/99% detection confidence.

    Example:\n
      pathogeniq plan --prevalence 0.0001\n
      pathogeniq plan --prevalence 0.0001 --depth 5000000
    """
    from pathogeniq.planning.calculator import SequencingPlan, plan_report
    from rich.console import Console
    from rich.table import Table

    console = Console()

    try:
        p = SequencingPlan(
            prevalence=prevalence, depth=depth, min_reads=min_reads,
            cost_per_million_reads=cost_per_million_reads, sample_prep_cost=sample_prep_cost,
        )
        report = plan_report(p)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1)

    if report["current"]:
        console.print(
            f"\n[bold]At depth {depth:,}:[/bold]  "
            f"sensitivity={report['current']['sensitivity']*100:.1f}%  "
            f"cost=${report['current']['cost']:.2f}"
        )

    table = Table(title="Required Depth / Cost by Confidence Target", show_lines=True)
    table.add_column("Confidence", justify="right")
    table.add_column("Required Depth", justify="right")
    table.add_column("Cost ($)", justify="right")
    for target, vals in report["targets"].items():
        table.add_row(f"{float(target)*100:.0f}%", f"{vals['depth']:,}", f"{vals['cost']:.2f}")
    console.print(table)


if __name__ == "__main__":
    cli()
