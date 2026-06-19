"""
pipeline/runner.py
Orchestrates the full PathogenIQ pipeline end-to-end.

Flow:
  1. Ingest      — load Kraken2 reports or raw FASTQs
  2. Filter      — remove rare/low-prevalence taxa
  3. Graph       — build FDR-corrected co-occurrence network
  4. SBM         — fit stochastic block model, find communities
  5. Novelty     — score each sample for anomalous profiles
  6. Risk        — compute composite risk score per sample
  6.2 Outbreak   — sample-to-sample Bray-Curtis similarity, dendrogram clustering
  6.5 Temporal   — baseline z-score, CUSUM changepoint, trend + forecast
  6.6 External   — correlate temporal alerts against an external signal feed
  7. Report      — save JSON + HTML reports (with temporal charts)
  8. Alert       — print/return alerts for HIGH/CRITICAL/trending samples
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class PipelineConfig:
    # Graph
    spearman_threshold: float = 0.45
    fdr_alpha: float = 0.05
    min_prevalence: float = 0.1
    min_total_reads: int = 50
    rank_bump: bool = False         # bump species below threshold up to genus instead of dropping
    # SBM
    max_k: int = 12
    sbm_n_init: int = 10
    # Novelty
    anomaly_contamination: float = 0.1
    # Risk
    alert_threshold: float = 0.6
    # Temporal
    temporal_window: int = 8        # weeks of history for baseline
    cusum_k: float = 0.5            # CUSUM allowance parameter
    cusum_h: float = 4.0            # CUSUM alert threshold
    db_path: str = ""               # SQLite history DB path (default: ~/.pathogeniq/history.db)
    run_date: str = ""              # ISO date for this run (default: today)
    # Reporting
    output_dir: str = "./reports"
    # Alerting
    alert_email: bool = False       # send email alert on HIGH/CRITICAL
    alert_slack: bool = False       # send Slack alert on HIGH/CRITICAL
    # VQ-VAE sequence embedding
    use_vqvae: bool = False         # enable sequence-level novelty detection
    vqvae_model_path: str = ""      # path to trained .pt checkpoint

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        alerting = cfg.get("alerting", {})
        vqvae = cfg.get("vqvae", {})
        return cls(
            spearman_threshold=cfg.get("graph", {}).get("spearman_threshold", 0.45),
            fdr_alpha=cfg.get("graph", {}).get("fdr_alpha", 0.05),
            min_prevalence=cfg.get("graph", {}).get("min_prevalence", 0.1),
            min_total_reads=cfg.get("graph", {}).get("min_total_reads", 50),
            rank_bump=cfg.get("graph", {}).get("rank_bump", False),
            max_k=cfg.get("sbm", {}).get("max_k", 12),
            sbm_n_init=cfg.get("sbm", {}).get("n_init", 10),
            alert_threshold=cfg.get("risk", {}).get("alert_threshold", 0.6),
            output_dir=cfg.get("reporting", {}).get("output_dir", "./reports"),
            alert_email=alerting.get("email", {}).get("enabled", False),
            alert_slack=alerting.get("slack", {}).get("enabled", False),
            use_vqvae=vqvae.get("enabled", False),
            vqvae_model_path=vqvae.get("model_path", ""),
        )


def run(
    input_path: str | Path,
    config: PipelineConfig | None = None,
    rank: str = "G",
    run_characterization: bool = False,
    quiet: bool = False,
    store: object | None = None,
    format: str = "auto",
    actor: str | None = None,
    external_signals: str | Path | None = None,
) -> dict:
    """
    Run the full pipeline on a directory of classifier reports or a count matrix TSV.

    Args:
        input_path: directory with classifier report files OR path to a TSV/CSV count matrix
        config: PipelineConfig (uses defaults if None)
        rank: taxonomic rank — 'G'=genus, 'S'=species
        run_characterization: if True, runs ESMFold on flagged novel pathogens
        quiet: suppress progress output
        store: optional history store implementing record_run/get_site_history/
            all_sites/run_count (e.g. saas.db.PostgresOrgStore). Defaults to the
            SQLite-backed TimeSeriesStore at cfg.db_path/DEFAULT_DB.
        format: classifier report format for directory input — "auto" (content-
            sniffed per file), "kraken2", "bracken", or "metaphlan". Ignored for
            count-matrix (file) input.
        actor: identifier of who/what triggered this run (CLI user, dashboard
            upload, etc.), recorded in the provenance manifest and history store.
        external_signals: optional path to a site/date/value CSV or TSV
            (e.g. qPCR, ddPCR, case counts) to cross-validate temporal alerts
            against. Skipped if None.

    Returns:
        dict with keys: sampleset, sbm_result, risk_scores, alerts, report_paths
    """
    from rich.console import Console
    from rich.progress import track

    cfg = config or PipelineConfig()
    console = Console(quiet=quiet)
    input_path = Path(input_path)

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    console.rule("[bold]PathogenIQ[/bold] — Step 1: Ingestion")
    from ..ingestion.reader import load_sample_directory, load_count_matrix, filter_taxa

    if input_path.is_dir():
        console.print(f"Loading classifier reports from [cyan]{input_path}[/cyan]")
        sampleset = load_sample_directory(input_path, rank=rank, format=format)
        ingested_files: list[Path] = []
        for s in sampleset.samples:
            if s.kraken_report:
                ingested_files.append(s.kraken_report)
            if s.kraken_report_2:
                ingested_files.append(s.kraken_report_2)
        formats_seen = {s.source_format for s in sampleset.samples}
        classifier_format = formats_seen.pop() if len(formats_seen) == 1 else "mixed"
    else:
        console.print(f"Loading count matrix from [cyan]{input_path}[/cyan]")
        sampleset = load_count_matrix(input_path)
        ingested_files = [input_path]
        classifier_format = "count_matrix"

    console.print(f"  {len(sampleset.samples)} samples, {sampleset.taxa_matrix.shape[0]} taxa")
    raw_sampleset = sampleset   # snapshot before Step 2's cohort-level filtering — rarefaction needs per-sample raw counts

    if external_signals:
        ingested_files.append(Path(external_signals))

    # ── Step 2: Filter ────────────────────────────────────────────────────────
    console.rule("Step 2: Taxa Filtering")
    sampleset = filter_taxa(
        sampleset,
        min_prevalence=cfg.min_prevalence,
        min_total_reads=cfg.min_total_reads,
        rank_bump=cfg.rank_bump,
    )
    console.print(f"  After filtering: {sampleset.taxa_matrix.shape[0]} taxa retained")

    # ── Step 2.5: Rarefaction Curves ─────────────────────────────────────────
    console.rule("Step 2.5: Rarefaction Curves")
    rarefaction_data: dict = {}
    try:
        from ..planning.rarefaction import rarefaction_for_sampleset
        curves = rarefaction_for_sampleset(raw_sampleset)
        rarefaction_data = {
            name: {
                "depths": c.depths, "richness": c.richness,
                "total_reads": c.total_reads, "observed_richness": c.observed_richness,
            }
            for name, c in curves.items()
        }
        console.print(f"  Computed rarefaction curves for {len(rarefaction_data)} samples")
    except Exception as exc:
        console.print(f"  [yellow]Rarefaction skipped: {exc}[/yellow]")

    n_taxa = sampleset.taxa_matrix.shape[0]
    sbm_result = None

    if n_taxa < 3:
        console.print("[yellow]Too few taxa for community analysis — skipping graph/SBM[/yellow]")
    else:
        # ── Step 3: Build co-occurrence graph ─────────────────────────────────
        console.rule("Step 3: Co-occurrence Graph (FDR-corrected)")
        from ..community.graph import build_cooccurrence_graph
        adj, taxa_names, rho_matrix = build_cooccurrence_graph(
            sampleset,
            spearman_threshold=cfg.spearman_threshold,
            fdr_alpha=cfg.fdr_alpha,
        )

        # ── Step 4: Fit SBM ───────────────────────────────────────────────────
        console.rule("Step 4: Stochastic Block Model")
        from ..community.sbm import fit_sbm
        from ..ingestion.reader import SampleSet

        # Build a minimal sampleset with just the filtered taxa for SBM
        sbm_sampleset = SampleSet(
            samples=sampleset.samples,
            taxa_matrix=sampleset.taxa_matrix.loc[taxa_names],
            relative_abundance=sampleset.relative_abundance.loc[taxa_names],
        )
        sbm_result = fit_sbm(
            adj,
            taxa_names=taxa_names,
            max_k=min(cfg.max_k, n_taxa - 1),
            n_init=cfg.sbm_n_init,
        )
        console.print(f"  Communities found: K={sbm_result.k}")

    # ── Step 4.5: Build graph_data dict for report ─────────────────────────────
    graph_data: dict = {"nodes": [], "edges": []}
    if sbm_result is not None:
        rel = sampleset.relative_abundance
        for i, taxon in enumerate(taxa_names):
            mean_abund = float(rel.loc[taxon].mean()) if taxon in rel.index else 0.0
            graph_data["nodes"].append({
                "id": taxon,
                "community": int(sbm_result.labels[i]),
                "mean_abundance": round(mean_abund, 6),
            })
        for i in range(len(taxa_names)):
            for j in range(i + 1, len(taxa_names)):
                if adj[i, j] == 1:
                    graph_data["edges"].append({
                        "source": taxa_names[i],
                        "target": taxa_names[j],
                        "rho": round(float(rho_matrix[i, j]), 4),
                    })

    # ── Step 5: Novelty detection ─────────────────────────────────────────────
    console.rule("Step 5: Novelty Detection")
    from ..novelty.detector import compute_novelty_scores
    novelty_scores = compute_novelty_scores(
        sampleset,
        contamination=cfg.anomaly_contamination,
    )
    for name, score in sorted(novelty_scores.items(), key=lambda x: -x[1]):
        if score > 0.5:
            console.print(f"  [yellow]Novelty flag:[/yellow] {name} (score={score:.3f})")

    # ── Step 5.5: Unsupervised clustering + differential abundance ────────────
    console.rule("Step 5.5: Sample Clustering & Differential Abundance")
    cluster_data: dict = {"n_clusters": 0, "assignments": {}, "silhouette": 0.0, "differential": []}
    try:
        from ..clustering.cluster import cluster_samples
        from ..clustering.differential import differential_abundance

        cluster_result = cluster_samples(sampleset)
        console.print(
            f"  {cluster_result.n_clusters} clusters found "
            f"(silhouette={cluster_result.silhouette:.3f})"
        )

        diff_results = differential_abundance(sampleset, cluster_result.labels)
        n_sig = sum(1 for r in diff_results if r.significant)
        console.print(f"  Differential taxa (BH p<0.05): {n_sig} / {len(diff_results)}")

        cluster_data = {
            "n_clusters": cluster_result.n_clusters,
            "silhouette": cluster_result.silhouette,
            "assignments": cluster_result.labels,
            "differential": [
                {
                    "taxon": r.taxon,
                    "H_stat": r.H_stat,
                    "p_value": r.p_value,
                    "p_adj": r.p_adj,
                    "significant": r.significant,
                    "cluster_means": r.cluster_means,
                    "fold_change": r.fold_change,
                }
                for r in diff_results
            ],
        }
    except Exception as exc:
        console.print(f"  [yellow]Clustering skipped: {exc}[/yellow]")

    # Build abundance matrix (top 50 taxa by mean abundance, all samples)
    abundance_matrix: dict = {"taxa": [], "samples": [], "values": []}
    try:
        rel_full = sampleset.relative_abundance  # taxa × samples
        top_taxa = rel_full.mean(axis=1).nlargest(50).index.tolist()
        abundance_matrix = {
            "taxa": top_taxa,
            "samples": list(rel_full.columns),
            "values": rel_full.loc[top_taxa].round(6).values.tolist(),
            "cluster_assignments": cluster_data.get("assignments", {}),
        }
    except Exception:
        pass

    # ── Step 5.75: VQ-VAE sequence embedding (optional) ──────────────────────
    vqvae_results: dict = {}
    if cfg.use_vqvae:
        console.rule("Step 5.75: VQ-VAE Sequence Embedding")
        from ..embedding.embedder import score_sequences, load_model
        from pathlib import Path as _P

        model_path = _P(cfg.vqvae_model_path) if cfg.vqvae_model_path else None
        if model_path is None:
            from ..embedding.embedder import DEFAULT_MODEL_PATH
            model_path = DEFAULT_MODEL_PATH

        if not model_path.exists():
            console.print(
                f"  [yellow]VQ-VAE model not found at {model_path}[/yellow]\n"
                "  Run [bold]pathogeniq train-embedder <reference.fasta>[/bold] first. Skipping."
            )
        else:
            # Collect sequences from sampleset if available (requires raw FASTA input)
            raw_seqs: dict[str, str] = getattr(sampleset, "raw_sequences", {}) or {}
            if not raw_seqs:
                console.print("  [dim]No raw sequences in sampleset — VQ-VAE skipped "
                              "(provide FASTA input or populate sampleset.raw_sequences)[/dim]")
            else:
                console.print(f"  Scoring {len(raw_seqs)} sequences ...")
                vqvae_results = score_sequences(raw_seqs, model_path=model_path)
                # Blend VQ-VAE novelty into isolation-forest novelty scores (max fusion)
                for taxon, emb in vqvae_results.items():
                    if emb.is_novel:
                        console.print(
                            f"  [red]Novel sequence:[/red] {taxon} "
                            f"(recon_loss={emb.reconstruction_loss:.4f}, "
                            f"novelty={emb.novelty_score:.3f})"
                        )
                # Propagate to sample-level novelty scores by taking the max
                for sample in sampleset.samples:
                    rel_abund = sampleset.relative_abundance.get(sample.name)
                    if rel_abund is None:
                        continue
                    seq_novelty = max(
                        (vqvae_results[t].novelty_score
                         for t in rel_abund[rel_abund > 0.01].index
                         if t.split()[0] in vqvae_results or t in vqvae_results),
                        default=0.0,
                    )
                    if seq_novelty > novelty_scores.get(sample.name, 0.0):
                        novelty_scores[sample.name] = seq_novelty
                        console.print(
                            f"  [yellow]VQ-VAE elevated novelty:[/yellow] "
                            f"{sample.name} → {seq_novelty:.3f}"
                        )

    # ── Step 6: Risk scoring ──────────────────────────────────────────────────
    console.rule("Step 6: Risk Scoring")
    from ..scoring.risk import score_all_samples
    risk_scores = score_all_samples(
        sampleset,
        sbm_result=sbm_result,
        novelty_scores=novelty_scores,
        alert_threshold=cfg.alert_threshold,
    )

    alerts = [r for r in risk_scores if r.is_alert(cfg.alert_threshold)]
    for r in risk_scores:
        color = {"LOW": "green", "MODERATE": "yellow", "HIGH": "red", "CRITICAL": "bold red"}.get(r.level, "white")
        console.print(f"  [{color}]{r.level:8s}[/{color}]  {r.sample_name:<30s}  score={r.score:.3f}")

    # ── Step 6.2: Outbreak source similarity ──────────────────────────────────
    console.rule("Step 6.2: Outbreak Source Similarity")
    outbreak_data: dict = {"dendrogram": None, "clusters": []}
    try:
        from ..outbreak.similarity import (
            compute_sample_distances, build_dendrogram, identify_outbreak_clusters,
        )
        if len(sampleset.samples) >= 3:
            condensed, sample_names = compute_sample_distances(sampleset)
            dendro = build_dendrogram(condensed, sample_names)
            if dendro:
                flagged = {a.sample_name for a in alerts}
                clusters = identify_outbreak_clusters(dendro, flagged)
                outbreak_data = {
                    "dendrogram": {
                        "linkage": dendro.linkage_matrix,
                        "sample_order": dendro.sample_order,
                        "distance_matrix": dendro.distance_matrix,
                    },
                    "clusters": clusters,
                }
                n_flagged_clusters = sum(1 for c in clusters if c["contains_flagged"] and len(c["members"]) > 1)
                console.print(f"  {len(clusters)} cluster(s), {n_flagged_clusters} containing a flagged sample")
        else:
            console.print("[yellow]Too few samples for outbreak similarity — skipping[/yellow]")
    except Exception as exc:
        console.print(f"  [yellow]Outbreak similarity skipped: {exc}[/yellow]")

    # ── Step 6.5: Temporal analysis ───────────────────────────────────────────
    console.rule("Step 6.5: Temporal Analysis")
    from ..temporal.baseline import compute_baselines_all_sites
    from ..temporal.cusum import run_cusum_all_sites
    from ..temporal.trend import analyze_trends_all_sites

    if store is None:
        from ..temporal.store import TimeSeriesStore, DEFAULT_DB
        from pathlib import Path as _Path

        db_path = _Path(cfg.db_path) if cfg.db_path else DEFAULT_DB
        store = TimeSeriesStore(db_path)
        console.print(f"  History DB: {db_path} ({store.run_count()} previous runs)")
    else:
        console.print(f"  History store: {type(store).__name__} ({store.run_count()} previous runs)")

    # ── Provenance manifest ────────────────────────────────────────────────
    # Built here (not right after Step 1) because store_backend needs `store`
    # resolved to a concrete instance first.
    from .. import provenance
    from dataclasses import asdict as _asdict
    manifest = provenance.build_manifest(
        input_files=ingested_files,
        classifier_format=classifier_format,
        rank=rank,
        config_snapshot=_asdict(cfg),
        actor=actor,
        store_backend=type(store).__name__,
    )

    # Compute temporal signals BEFORE recording this run
    baselines = compute_baselines_all_sites(store, risk_scores, window=cfg.temporal_window)
    cusum_results = run_cusum_all_sites(store, risk_scores, window=cfg.temporal_window * 2,
                                        k=cfg.cusum_k, h=cfg.cusum_h)
    trend_results = analyze_trends_all_sites(store, risk_scores, window=cfg.temporal_window * 2)

    from datetime import datetime, timezone
    current_date = cfg.run_date or datetime.now(timezone.utc).date().isoformat()

    # ── Step 6.6: External signal validation ──────────────────────────────────
    external_validation_results: dict = {}
    if external_signals:
        console.rule("Step 6.6: External Signal Validation")
        from ..external.signals import load_external_signals, validate_all_sites
        from dataclasses import asdict as _ext_asdict
        try:
            ext_df = load_external_signals(external_signals)
            external_validation_results = validate_all_sites(
                ext_df, store, risk_scores, run_date=current_date, window=cfg.temporal_window * 2,
            )
            for site, result in external_validation_results.items():
                console.print(f"  {site:<30s}  {result.summary}")
        except ValueError as exc:
            console.print(f"  [yellow]External signal validation skipped: {exc}[/yellow]")
    for rs in risk_scores:
        ext_result = external_validation_results.get(rs.sample_name)
        rs.external_validation = _ext_asdict(ext_result) if ext_result else None

    # Print temporal summary
    temporal_alerts = []
    for rs in risk_scores:
        name = rs.sample_name
        bl = baselines.get(name)
        cu = cusum_results.get(name)
        tr = trend_results.get(name)

        flags = []
        if bl and bl.is_anomaly:
            flags.append(f"[yellow]BASELINE+{bl.pct_above_baseline:.0f}%[/yellow]")
        if cu and cu.alert:
            flags.append(f"[red]CUSUM({cu.signal_strength})[/red]")
        if tr and tr.trend == "increasing" and tr.is_significant:
            flags.append(f"[orange1]TREND↑(τ={tr.tau:.2f})[/orange1]")

        flag_str = "  ".join(flags) if flags else "[dim]stable[/dim]"
        console.print(f"  {name:<30s}  z={bl.z_score:+.2f}  CUSUM={cu.cusum_upper:.2f}  {flag_str}")

        if flags:
            temporal_alerts.append(name)

    # Record this run to history
    store.record_run(risk_scores, sampleset, run_date=current_date, input_path=str(input_path), rank=rank,
                      manifest=manifest)
    console.print(f"  Run recorded to history (total runs: {store.run_count()})")

    # Add temporal alerts to main alert list
    for name in temporal_alerts:
        if not any(a.sample_name == name for a in alerts):
            matching = next((r for r in risk_scores if r.sample_name == name), None)
            if matching:
                alerts.append(matching)

    # ── Step 6.7: Species-level lineage context ───────────────────────────────
    console.rule("Step 6.7: Lineage Context (species rank)")
    from ..lineage.detector import load_species_from_report, lineage_annotations_to_dict
    lineage_by_sample: dict[str, list[dict]] = {}
    for sample in sampleset.samples:
        if sample.kraken_report:
            annotations = load_species_from_report(sample.kraken_report, fmt=sample.source_format)
            lineage_by_sample[sample.name] = lineage_annotations_to_dict(annotations)
            if annotations:
                top = annotations[0]
                console.print(
                    f"  {sample.name:<30s}  species: {top.species} "
                    f"({top.abundance*100:.1f}%)"
                )

    # ── Step 6.8: AMR annotation ──────────────────────────────────────────────
    console.rule("Step 6.8: AMR Annotation")
    from ..amr.annotator import annotate_amr, amr_annotations_to_dict
    for rs in risk_scores:
        amr = annotate_amr(rs.detected_pathogens)
        rs.amr_annotations = amr_annotations_to_dict(amr)
        rs.lineage_annotations = lineage_by_sample.get(rs.sample_name, [])
        if amr:
            top = amr[0]
            console.print(
                f"  {rs.sample_name:<30s}  AMR: {top.genus} "
                f"[{top.who_priority.upper()}] — "
                f"{', '.join(r['drug_class'] for r in top.resistances[:2])}"
            )

    # ── Step 7: Characterization (optional) ───────────────────────────────────
    characterization_results = {}
    if run_characterization and alerts:
        console.rule("Step 7: AlphaFold Characterization")
        from ..characterization.alphafold import characterize_novel_pathogen
        flagged_taxa = set()
        for alert in alerts:
            for p in alert.detected_pathogens[:2]:
                flagged_taxa.add(p["taxon"])
        for taxon in flagged_taxa:
            console.print(f"  Characterizing [cyan]{taxon}[/cyan] via ESMFold...")
            preds = characterize_novel_pathogen(taxon)
            characterization_results[taxon] = preds
            for pred in preds:
                console.print(f"    pLDDT={pred.confidence:.1f}  flags={pred.virulence_flags}")

    # ── Step 8: Report ────────────────────────────────────────────────────────
    console.rule("Step 8: Reporting")
    import json
    from ..reporting.report import to_dict, save_html
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"input": str(input_path), "rank": rank, "n_samples": len(sampleset.samples)}
    json_path = out_dir / "report.json"
    html_path = out_dir / "report.html"
    provenance_path = out_dir / "provenance.json"

    report_dict = to_dict(risk_scores, meta, baselines, cusum_results, trend_results,
                          graph_data, cluster_data, abundance_matrix, outbreak_data,
                          rarefaction_data)
    report_content_hash = provenance.compute_report_content_hash(report_dict["samples"])
    manifest["report_content_hash"] = report_content_hash
    report_dict["metadata"]["provenance"] = manifest

    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    console.print(f"  JSON report → {json_path}")
    with open(provenance_path, "w") as f:
        json.dump(manifest, f, indent=2)
    console.print(f"  Provenance manifest → {provenance_path}")

    save_html(risk_scores, html_path, meta=meta,
              baselines=baselines, cusum_results=cusum_results, trend_results=trend_results)

    if alerts:
        console.rule(f"[bold red]ALERTS ({len(alerts)} samples)[/bold red]")
        for r in alerts:
            console.print(f"  [bold red]ALERT[/bold red] {r.sample_name}: {r.level} (score={r.score:.3f})")
            for p in r.detected_pathogens[:3]:
                console.print(f"    ↳ {p['taxon']} — {p['disease']} (abundance={p['abundance']:.4f})")
    else:
        console.print("[green]No alerts — all samples within expected range[/green]")

    # ── Step 9: Alert dispatch ────────────────────────────────────────────────
    dispatch_results: dict = {}
    if alerts and (cfg.alert_email or cfg.alert_slack):
        console.rule("Step 9: Alert Dispatch")
        from ..alerting.dispatcher import AlertConfig, dispatch_alerts

        alert_cfg = AlertConfig.from_env()
        alert_cfg.email_enabled = cfg.alert_email
        alert_cfg.slack_enabled = cfg.alert_slack
        dispatch_results = dispatch_alerts(
            alerts, config=alert_cfg, run_date=cfg.run_date or ""
        )
    elif alerts:
        console.print(
            "  [dim]Alert dispatch disabled — set alert_email/alert_slack in config "
            "or PATHOGENIQ_SMTP_USER / PATHOGENIQ_SLACK_WEBHOOK env vars[/dim]"
        )

    return {
        "sampleset": sampleset,
        "sbm_result": sbm_result,
        "risk_scores": risk_scores,
        "novelty_scores": novelty_scores,
        "alerts": alerts,
        "characterization": characterization_results,
        "temporal": {
            "baselines": baselines,
            "cusum": cusum_results,
            "trends": trend_results,
            "store": store,
        },
        "report_paths": {"json": str(json_path), "html": str(html_path), "provenance": str(provenance_path)},
        "manifest": manifest,
        "vqvae": vqvae_results,
        "dispatch": dispatch_results,
    }
