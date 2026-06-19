"""
tests/test_ingestion_formats.py
Classifier-agnostic ingestion: format sniffing, per-format parsing, and
end-to-end directory loading for Bracken and MetaPhlAn4 reports.

Fully self-contained via tmp_path — does not depend on tests/fixtures/,
which is missing the pre-existing sample_a.report/sample_b.report fixtures.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pathogeniq.ingestion.formats import (
    sniff_format,
    load_bracken_report,
    load_metaphlan_report,
    load_report,
)
from pathogeniq.ingestion.reader import load_sample_directory, filter_taxa


KRAKEN2_REPORT = """\
 10.00\t400\t400\tG\t561\t  Escherichia
 20.00\t200\t200\tG\t816\t  Bacteroides
 30.00\t380\t380\tS\t562\t  Escherichia coli
"""

BRACKEN_REPORT = """\
name\ttaxonomy_id\ttaxonomy_lvl\tkraken_assigned_reads\tadded_reads\tnew_est_reads\tfraction_total_reads
Escherichia\t561\tG\t4000\t200\t4200\t0.42
Bacteroides\t816\tG\t2000\t100\t2100\t0.21
Escherichia coli\t562\tS\t3800\t150\t3950\t0.395
Bacteroides fragilis\t817\tS\t1900\t90\t1990\t0.199
"""

METAPHLAN_REPORT = """\
#mpa_v31_CHOCOPhlAn_201901
#clade_name\tNCBI_tax_id\trelative_abundance\tadditional_species
k__Bacteria\t2\t100.0\t
k__Bacteria|p__Proteobacteria|c__Gammaproteobacteria|o__Enterobacterales|f__Enterobacteriaceae|g__Escherichia\t2|1224|1236|91347|543|561\t45.0\t
k__Bacteria|p__Proteobacteria|c__Gammaproteobacteria|o__Enterobacterales|f__Enterobacteriaceae|g__Escherichia|s__Escherichia_coli\t2|1224|1236|91347|543|561|562\t30.0\t
k__Bacteria|p__Bacteroidetes|c__Bacteroidia|o__Bacteroidales|f__Bacteroidaceae|g__Bacteroides\t2|976|200643|171549|815|816\t55.0\t
k__Bacteria|p__Bacteroidetes|c__Bacteroidia|o__Bacteroidales|f__Bacteroidaceae|g__Bacteroides|s__Bacteroides_fragilis\t2|976|200643|171549|815|816|817\t40.0\t
"""


# ── sniff_format ─────────────────────────────────────────────────────────────

def test_sniff_kraken2(tmp_path):
    p = tmp_path / "sample.report"
    p.write_text(KRAKEN2_REPORT)
    assert sniff_format(p) == "kraken2"


def test_sniff_bracken(tmp_path):
    p = tmp_path / "sample.bracken"
    p.write_text(BRACKEN_REPORT)
    assert sniff_format(p) == "bracken"


def test_sniff_metaphlan(tmp_path):
    p = tmp_path / "sample_profile.tsv"
    p.write_text(METAPHLAN_REPORT)
    assert sniff_format(p) == "metaphlan"


def test_sniff_tsv_extension_with_bracken_content_is_bracken(tmp_path):
    """.tsv is used by Bracken, MetaPhlAn, and plain count matrices alike —
    sniffing must go by content, not extension."""
    p = tmp_path / "ambiguous.tsv"
    p.write_text(BRACKEN_REPORT)
    assert sniff_format(p) == "bracken"


def test_sniff_metaphlan_without_comment_header(tmp_path):
    """A MetaPhlAn profile with its leading '#' comment stripped still sniffs
    correctly via the bare 'clade_name' header."""
    p = tmp_path / "no_comment.tsv"
    lines = METAPHLAN_REPORT.splitlines()
    body = [lines[1].lstrip("#")] + lines[2:]
    p.write_text("\n".join(body) + "\n")
    assert sniff_format(p) == "metaphlan"


# ── per-format parsing ───────────────────────────────────────────────────────

def test_load_bracken_report_genus(tmp_path):
    p = tmp_path / "sample.bracken"
    p.write_text(BRACKEN_REPORT)
    series = load_bracken_report(p, rank="G")
    assert series.name == "sample"
    assert series["Escherichia"] == 4200
    assert series["Bacteroides"] == 2100
    assert "Escherichia coli" not in series.index


def test_load_bracken_report_species(tmp_path):
    p = tmp_path / "sample.bracken"
    p.write_text(BRACKEN_REPORT)
    series = load_bracken_report(p, rank="S")
    assert series["Escherichia coli"] == 3950
    assert series["Bacteroides fragilis"] == 1990
    assert "Escherichia" not in series.index


def test_load_bracken_report_missing_columns_raises(tmp_path):
    p = tmp_path / "broken.bracken"
    p.write_text("name\ttaxonomy_id\ttaxonomy_lvl\nEscherichia\t561\tG\n")
    with pytest.raises(ValueError):
        load_bracken_report(p, rank="G")


def test_load_metaphlan_report_genus(tmp_path):
    p = tmp_path / "sample_profile.tsv"
    p.write_text(METAPHLAN_REPORT)
    series = load_metaphlan_report(p, rank="G")
    assert series.name == "sample_profile"
    assert series["Escherichia"] == 45.0
    assert series["Bacteroides"] == 55.0
    assert "Escherichia_coli" not in series.index


def test_load_metaphlan_report_species(tmp_path):
    p = tmp_path / "sample_profile.tsv"
    p.write_text(METAPHLAN_REPORT)
    series = load_metaphlan_report(p, rank="S")
    assert series["Escherichia_coli"] == 30.0
    assert series["Bacteroides_fragilis"] == 40.0
    assert "Escherichia" not in series.index


def test_load_metaphlan_report_no_data_rows_raises(tmp_path):
    p = tmp_path / "empty_profile.tsv"
    p.write_text("#mpa_v31\n#clade_name\tNCBI_tax_id\trelative_abundance\n")
    with pytest.raises(ValueError):
        load_metaphlan_report(p, rank="G")


# ── load_report dispatch ─────────────────────────────────────────────────────

def test_load_report_dispatches_kraken2(tmp_path):
    p = tmp_path / "sample.report"
    p.write_text(KRAKEN2_REPORT)
    series, fmt = load_report(p, rank="G")
    assert fmt == "kraken2"
    assert series["Escherichia"] == 400


def test_load_report_dispatches_bracken(tmp_path):
    p = tmp_path / "sample.bracken"
    p.write_text(BRACKEN_REPORT)
    series, fmt = load_report(p, rank="G")
    assert fmt == "bracken"
    assert series["Escherichia"] == 4200


def test_load_report_dispatches_metaphlan(tmp_path):
    p = tmp_path / "sample_profile.tsv"
    p.write_text(METAPHLAN_REPORT)
    series, fmt = load_report(p, rank="G")
    assert fmt == "metaphlan"
    assert series["Escherichia"] == 45.0


def test_load_report_explicit_format_overrides_sniffing(tmp_path):
    """fmt='bracken' should skip sniffing entirely and parse as Bracken even
    if the file is named like a Kraken2 report."""
    p = tmp_path / "sample.report"
    p.write_text(BRACKEN_REPORT)
    series, fmt = load_report(p, rank="G", fmt="bracken")
    assert fmt == "bracken"
    assert series["Escherichia"] == 4200


# ── end-to-end load_sample_directory ────────────────────────────────────────

def _write_bracken_pair(directory: Path):
    siteA = """\
name\ttaxonomy_id\ttaxonomy_lvl\tkraken_assigned_reads\tadded_reads\tnew_est_reads\tfraction_total_reads
Escherichia\t561\tG\t4000\t200\t4200\t0.6
Bacteroides\t816\tG\t1500\t100\t1600\t0.23
Salmonella\t590\tG\t800\t50\t850\t0.12
"""
    siteB = """\
name\ttaxonomy_id\ttaxonomy_lvl\tkraken_assigned_reads\tadded_reads\tnew_est_reads\tfraction_total_reads
Escherichia\t561\tG\t3000\t150\t3150\t0.5
Bacteroides\t816\tG\t2000\t120\t2120\t0.34
Salmonella\t590\tG\t60\t5\t65\t0.01
"""
    (directory / "siteA.bracken").write_text(siteA)
    (directory / "siteB.bracken").write_text(siteB)


def test_load_sample_directory_bracken_only(tmp_path):
    _write_bracken_pair(tmp_path)
    sampleset = load_sample_directory(tmp_path, rank="G")

    assert sorted(sampleset.sample_names) == ["siteA", "siteB"]
    assert all(s.source_format == "bracken" for s in sampleset.samples)
    assert "Escherichia" in sampleset.taxa_matrix.index

    filtered = filter_taxa(sampleset, min_prevalence=0.5, min_total_reads=100)
    assert "Escherichia" in filtered.taxa_matrix.index
    assert filtered.relative_abundance.shape[1] == 2


def _write_metaphlan_pair(directory: Path):
    siteA = """\
#mpa_v31_CHOCOPhlAn_201901
#clade_name\tNCBI_tax_id\trelative_abundance\tadditional_species
k__Bacteria|p__Proteobacteria|c__Gammaproteobacteria|o__Enterobacterales|f__Enterobacteriaceae|g__Escherichia\t2|1224|1236|91347|543|561\t60.0\t
k__Bacteria|p__Bacteroidetes|c__Bacteroidia|o__Bacteroidales|f__Bacteroidaceae|g__Bacteroides\t2|976|200643|171549|815|816\t30.0\t
k__Bacteria|p__Proteobacteria|c__Gammaproteobacteria|o__Enterobacterales|f__Enterobacteriaceae|g__Salmonella\t2|1224|1236|91347|543|590\t10.0\t
"""
    siteB = """\
#mpa_v31_CHOCOPhlAn_201901
#clade_name\tNCBI_tax_id\trelative_abundance\tadditional_species
k__Bacteria|p__Proteobacteria|c__Gammaproteobacteria|o__Enterobacterales|f__Enterobacteriaceae|g__Escherichia\t2|1224|1236|91347|543|561\t45.0\t
k__Bacteria|p__Bacteroidetes|c__Bacteroidia|o__Bacteroidales|f__Bacteroidaceae|g__Bacteroides\t2|976|200643|171549|815|816\t50.0\t
k__Bacteria|p__Proteobacteria|c__Gammaproteobacteria|o__Enterobacterales|f__Enterobacteriaceae|g__Salmonella\t2|1224|1236|91347|543|590\t5.0\t
"""
    (directory / "siteA_profile.tsv").write_text(siteA)
    (directory / "siteB_profile.tsv").write_text(siteB)


def test_load_sample_directory_metaphlan_only(tmp_path):
    _write_metaphlan_pair(tmp_path)
    sampleset = load_sample_directory(tmp_path, rank="G")

    assert sorted(sampleset.sample_names) == ["siteA_profile", "siteB_profile"]
    assert all(s.source_format == "metaphlan" for s in sampleset.samples)
    assert "Escherichia" in sampleset.taxa_matrix.index

    filtered = filter_taxa(sampleset, min_prevalence=0.5, min_total_reads=50)
    assert "Escherichia" in filtered.taxa_matrix.index
    assert filtered.relative_abundance.shape[1] == 2


def test_load_sample_directory_warns_on_mixed_formats(tmp_path, capsys):
    """Mixing classifier formats in one run is allowed but flagged. Both files
    use the .tsv extension so they're picked up by the same fallback glob
    pattern, even though their content (and therefore sniffed format) differs."""
    (tmp_path / "siteA.tsv").write_text(
        "name\ttaxonomy_id\ttaxonomy_lvl\tkraken_assigned_reads\tadded_reads\t"
        "new_est_reads\tfraction_total_reads\n"
        "Escherichia\t561\tG\t4000\t200\t4200\t0.6\n"
    )
    (tmp_path / "siteB.tsv").write_text(
        "#mpa_v31_CHOCOPhlAn_201901\n"
        "#clade_name\tNCBI_tax_id\trelative_abundance\tadditional_species\n"
        "k__Bacteria|p__Proteobacteria|c__Gammaproteobacteria|o__Enterobacterales|"
        "f__Enterobacteriaceae|g__Escherichia\t2|1224|1236|91347|543|561\t45.0\t\n"
    )

    sampleset = load_sample_directory(tmp_path, rank="G")
    assert {s.source_format for s in sampleset.samples} == {"bracken", "metaphlan"}
    captured = capsys.readouterr()
    assert "mixing classifier formats" in captured.out.lower()
