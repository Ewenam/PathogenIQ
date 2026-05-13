#!/usr/bin/env python3
"""
fetch_metadata.py — PathogenIQ SRA Metadata Fetcher
=====================================================
Automatically extracts SRR/ERR/DRR accession IDs from your sample filenames,
pulls metadata from NCBI SRA (title, study, location, date, isolation source),
geocodes location strings to lat/lon via OpenStreetMap Nominatim, and writes:

  1. A formatted summary table to the terminal
  2. site_locations.json  — auto-populated for the PathogenIQ map view
  3. metadata.tsv         — sample × attribute table for downstream analysis

Usage:
    python3 fetch_metadata.py

Set INPUT_DIR to your Kraken2 reports folder (or set it to the same value as
INPUT in run.py).  Everything else is automatic.

Auto-switch to venv if dependencies are missing:
"""
import sys, os
from pathlib import Path

_VENV_CANDIDATES = [
    Path(__file__).parent.parent / ".venv" / "bin" / "python3",
    Path(__file__).parent / "venv" / "bin" / "python3",
]
try:
    import requests
except ImportError:
    for _venv_py in _VENV_CANDIDATES:
        if _venv_py.exists():
            os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)
    sys.exit("ERROR: 'requests' not found. Run: pip install requests")

# ─────────────────────────────────────────────
#  SET THIS — same as INPUT in run.py
# ─────────────────────────────────────────────
INPUT_DIR = "/home/users/razumah1/Desktop/AAB Project/data/reports"

# Where to write outputs (same folder as PathogenIQ dashboard)
SITE_LOCATIONS_JSON = Path(__file__).parent / "pathogeniq" / "dashboard" / "site_locations.json"
METADATA_TSV        = Path(__file__).parent / "reports" / "metadata.tsv"

# NCBI API key (optional — increases rate limit from 3 to 10 req/s)
# Get one free at https://www.ncbi.nlm.nih.gov/account/
NCBI_API_KEY = ""

# ─────────────────────────────────────────────
#  DO NOT EDIT BELOW THIS LINE
# ─────────────────────────────────────────────
import re
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
ACC_PATTERN = re.compile(r'((?:SRR|ERR|DRR|SRS|ERS|DRS)\d{6,})')

# Built-in fallback coordinates for common US states + select counties.
# Used when Nominatim is unreachable (e.g. restricted networks).
_US_FALLBACK: dict[str, tuple[float, float]] = {
    # States (centroid)
    "alabama": (32.806, -86.791), "alaska": (61.370, -152.404),
    "arizona": (33.729, -111.431), "arkansas": (34.969, -92.373),
    "california": (36.116, -119.682), "colorado": (39.060, -105.311),
    "connecticut": (41.597, -72.755), "delaware": (39.318, -75.507),
    "florida": (27.766, -81.687), "georgia": (33.040, -83.643),
    "hawaii": (21.094, -157.498), "idaho": (44.240, -114.478),
    "illinois": (40.349, -88.986), "indiana": (39.849, -86.258),
    "iowa": (42.011, -93.210), "kansas": (38.526, -96.726),
    "kentucky": (37.668, -84.670), "louisiana": (31.169, -91.867),
    "maine": (44.693, -69.381), "maryland": (39.063, -76.802),
    "massachusetts": (42.230, -71.530), "michigan": (43.326, -84.536),
    "minnesota": (45.694, -93.900), "mississippi": (32.741, -89.678),
    "missouri": (38.456, -92.288), "montana": (46.921, -110.454),
    "nebraska": (41.125, -98.268), "nevada": (38.313, -117.055),
    "new hampshire": (43.452, -71.563), "new jersey": (40.298, -74.521),
    "new mexico": (34.840, -106.248), "new york": (42.165, -74.948),
    "north carolina": (35.630, -79.806), "north dakota": (47.528, -99.784),
    "ohio": (40.388, -82.764), "oklahoma": (35.565, -96.928),
    "oregon": (44.572, -122.071), "pennsylvania": (40.590, -77.209),
    "rhode island": (41.680, -71.511), "south carolina": (33.856, -80.945),
    "south dakota": (44.299, -99.438), "tennessee": (35.747, -86.692),
    "texas": (31.054, -97.563), "utah": (40.150, -111.862),
    "vermont": (44.045, -72.710), "virginia": (37.769, -78.169),
    "washington": (47.400, -121.490), "west virginia": (38.491, -80.954),
    "wisconsin": (43.073, -89.401), "wyoming": (42.755, -107.302),
    # Common Colorado counties
    "boulder county": (40.093, -105.371), "arapahoe county": (39.648, -104.834),
    "adams county": (39.871, -104.731), "denver county": (39.742, -104.984),
    "jefferson county": (39.535, -105.176), "larimer county": (40.661, -105.364),
}

_geocache: dict[str, tuple[float, float] | None] = {}


@dataclass
class SampleMeta:
    sample_name: str          # name as it appears in the PathogenIQ report
    accession: str            # SRR/ERR/DRR ID
    title: str = ""
    study: str = ""
    geo_loc_name: str = ""
    isolation_source: str = ""
    collection_date: str = ""
    lat: float | None = None
    lon: float | None = None
    extra: dict = field(default_factory=dict)


def _get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response | None:
    """GET with retries and polite rate-limiting."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(2)
    return None


def _extract_accessions(directory: Path) -> dict[str, str]:
    """
    Scan Kraken2 .report files in directory for SRR/ERR/DRR accession IDs.
    Returns {merged_sample_name: accession_id}.
    Paired-end _1/_2 files are collapsed to one entry (the merged name).
    """
    found: dict[str, str] = {}
    # Only look at .report files — skip .kraken.out, .fastq, etc.
    for f in sorted(directory.glob("*.report")):
        m = ACC_PATTERN.search(f.stem)
        if not m:
            continue
        acc = m.group(1)
        # Strip _1/_2 suffix to match the merged sample name from the pipeline
        paired_m = re.match(r'^(.+)_[12]$', f.stem)
        sample_name = paired_m.group(1) if paired_m else f.stem
        found[sample_name] = acc
    return found


def _ncbi_fetch_metadata(accessions: list[str]) -> dict[str, dict]:
    """Fetch BioSample attributes for a list of SRA accessions from NCBI eutils."""
    if not accessions:
        return {}

    params: dict = {"db": "sra", "term": " OR ".join(accessions),
                    "retmode": "json", "retmax": str(len(accessions) + 5)}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    r = _get(EUTILS + "esearch.fcgi", params)
    if not r:
        print("  ERROR: could not reach NCBI eutils — check your internet connection.")
        return {}

    ids = r.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        print("  WARNING: NCBI returned no records for these accessions.")
        return {}

    time.sleep(0.35)
    fetch_params: dict = {"db": "sra", "id": ",".join(ids), "retmode": "xml"}
    if NCBI_API_KEY:
        fetch_params["api_key"] = NCBI_API_KEY

    r2 = _get(EUTILS + "efetch.fcgi", fetch_params)
    if not r2:
        print("  ERROR: NCBI efetch failed.")
        return {}

    root = ET.fromstring(r2.text)
    results: dict[str, dict] = {}

    for pkg in root.findall(".//EXPERIMENT_PACKAGE"):
        run = pkg.find(".//RUN")
        if run is None:
            continue
        acc = run.get("accession", "")

        sample = pkg.find(".//SAMPLE")
        title = sample.findtext("TITLE", "").strip() if sample is not None else ""

        study = pkg.find(".//STUDY")
        study_title = ""
        if study is not None:
            desc = study.find("DESCRIPTOR")
            if desc is not None:
                study_title = (desc.findtext("STUDY_TITLE") or
                               desc.findtext("STUDY_ABSTRACT") or "").strip()

        attrs: dict[str, str] = {}
        for a in pkg.findall(".//SAMPLE_ATTRIBUTE"):
            tag = a.findtext("TAG", "").strip()
            val = a.findtext("VALUE", "").strip()
            if tag:
                attrs[tag] = val

        results[acc] = {
            "title": title,
            "study": study_title,
            "geo_loc_name": attrs.get("geo_loc_name", ""),
            "isolation_source": attrs.get("isolation_source", ""),
            "collection_date": attrs.get("collection_date", ""),
            "lat_lon": attrs.get("lat_lon", ""),
            "extra": {k: v for k, v in attrs.items()
                      if k not in {"geo_loc_name", "isolation_source",
                                   "collection_date", "lat_lon"}},
        }

    return results


def _parse_ncbi_latlon(lat_lon_str: str) -> tuple[float, float] | None:
    """Parse NCBI lat_lon format: '43.073 N 89.401 W' → (43.073, -89.401)."""
    m = re.match(
        r'([\d.]+)\s*([NS])\s+([\d.]+)\s*([EW])',
        lat_lon_str.strip(), re.IGNORECASE
    )
    if not m:
        return None
    lat = float(m.group(1)) * (1 if m.group(2).upper() == "N" else -1)
    lon = float(m.group(3)) * (1 if m.group(4).upper() == "E" else -1)
    return (round(lat, 5), round(lon, 5))


def _geocode_fallback(place: str) -> tuple[float, float] | None:
    """
    Lookup coordinates from the built-in US state/county table.
    Matches any word in the place string against the table keys (case-insensitive).
    """
    place_lower = place.lower()
    # Try longest match first (county names before state names)
    for key in sorted(_US_FALLBACK, key=len, reverse=True):
        if key in place_lower:
            return _US_FALLBACK[key]
    return None


def _geocode(place: str) -> tuple[float, float] | None:
    """
    Geocode a place name string.
    Tries Nominatim (OpenStreetMap) first; falls back to built-in US lookup table.
    NCBI format: 'USA: State, County'
    Results are cached for the session.
    """
    if not place:
        return None
    if place in _geocache:
        return _geocache[place]

    # Reformat 'USA: State, County' → 'County, State, USA' for Nominatim
    cleaned = re.sub(r'^[A-Z]+:\s*', '', place)
    parts = [p.strip() for p in cleaned.split(",")]
    query = ", ".join(reversed(parts)) if len(parts) > 1 else cleaned

    time.sleep(1.1)   # Nominatim rate limit: 1 req/s
    r = _get(NOMINATIM, {"q": query, "format": "json", "limit": "1"})
    if r:
        data = r.json()
        if data:
            result = (round(float(data[0]["lat"]), 5), round(float(data[0]["lon"]), 5))
            _geocache[place] = result
            return result

    # Nominatim unavailable (restricted network) — use built-in fallback table
    result = _geocode_fallback(place)
    _geocache[place] = result
    return result


def _build_label(meta: dict, acc: str) -> str:
    """Build a human-readable map label from metadata."""
    parts = []
    geo = meta.get("geo_loc_name", "")
    # Extract county/state from 'USA: Wisconsin' or 'USA: Colorado, Boulder County'
    geo_clean = re.sub(r'^[A-Z]+:\s*', '', geo)
    if geo_clean:
        parts.append(geo_clean)
    date = meta.get("collection_date", "")
    if date:
        parts.append(f"({date})")
    return ", ".join(parts) if parts else acc


def fetch_all(input_dir: Path) -> list[SampleMeta]:
    """Main entry point: scan directory, fetch NCBI metadata, geocode locations."""
    print(f"\nScanning {input_dir} for SRA accessions...")
    sample_to_acc = _extract_accessions(input_dir)

    if not sample_to_acc:
        print("  No SRR/ERR/DRR accessions found in filenames.")
        print("  Files must contain an accession ID like SRR12345678 in their name.")
        return []

    unique_accs = list(set(sample_to_acc.values()))
    print(f"  Found {len(sample_to_acc)} sample(s), {len(unique_accs)} unique accession(s):")
    for name, acc in sorted(sample_to_acc.items()):
        print(f"    {name:<40s} → {acc}")

    print(f"\nFetching NCBI metadata for {len(unique_accs)} accessions...")
    ncbi_data = _ncbi_fetch_metadata(unique_accs)
    print(f"  Retrieved {len(ncbi_data)} record(s) from NCBI.")

    print("\nGeocoding locations via OpenStreetMap Nominatim...")
    results: list[SampleMeta] = []

    for sample_name, acc in sorted(sample_to_acc.items()):
        meta_raw = ncbi_data.get(acc, {})
        sm = SampleMeta(
            sample_name=sample_name,
            accession=acc,
            title=meta_raw.get("title", ""),
            study=meta_raw.get("study", ""),
            geo_loc_name=meta_raw.get("geo_loc_name", ""),
            isolation_source=meta_raw.get("isolation_source", ""),
            collection_date=meta_raw.get("collection_date", ""),
            extra=meta_raw.get("extra", {}),
        )

        # Prefer explicit lat_lon from NCBI, fall back to geocoding geo_loc_name
        ncbi_ll = _parse_ncbi_latlon(meta_raw.get("lat_lon", ""))
        if ncbi_ll:
            sm.lat, sm.lon = ncbi_ll
            print(f"  {acc}: lat/lon from NCBI  ({sm.lat}, {sm.lon})")
        elif sm.geo_loc_name:
            coords = _geocode(sm.geo_loc_name)
            if coords:
                sm.lat, sm.lon = coords
                print(f"  {acc}: geocoded '{sm.geo_loc_name}' → ({sm.lat}, {sm.lon})")
            else:
                print(f"  {acc}: could not geocode '{sm.geo_loc_name}'")
        else:
            print(f"  {acc}: no location info available")

        results.append(sm)

    return results


def write_site_locations(results: list[SampleMeta], out_path: Path):
    """Write/update site_locations.json preserving existing non-SRA entries."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except Exception:
            pass

    # Keep comment/format keys and any manually-added entries
    updated = {k: v for k, v in existing.items() if k.startswith("_")}

    # Overwrite SRA-derived entries
    for sm in results:
        entry: dict = {}
        if sm.lat is not None and sm.lon is not None:
            entry["lat"] = sm.lat
            entry["lon"] = sm.lon
        label = _build_label(
            {"geo_loc_name": sm.geo_loc_name, "collection_date": sm.collection_date},
            sm.accession,
        )
        entry["label"] = label
        entry["accession"] = sm.accession
        if sm.collection_date:
            entry["collection_date"] = sm.collection_date
        if sm.isolation_source:
            entry["isolation_source"] = sm.isolation_source
        updated[sm.sample_name] = entry

    # Preserve manually-added entries that aren't SRA samples
    for k, v in existing.items():
        if k not in updated and not k.startswith("_"):
            updated[k] = v

    out_path.write_text(json.dumps(updated, indent=2))
    print(f"\nWrote site_locations.json → {out_path}")


def write_metadata_tsv(results: list[SampleMeta], out_path: Path):
    """Write a TSV with all metadata fields, usable for differential abundance labeling."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_extra_keys: list[str] = []
    for sm in results:
        for k in sm.extra:
            if k not in all_extra_keys:
                all_extra_keys.append(k)

    header = ["sample_name", "accession", "title", "study", "geo_loc_name",
              "isolation_source", "collection_date", "lat", "lon"] + all_extra_keys

    with open(out_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for sm in results:
            row = [
                sm.sample_name, sm.accession, sm.title, sm.study,
                sm.geo_loc_name, sm.isolation_source, sm.collection_date,
                str(sm.lat) if sm.lat is not None else "",
                str(sm.lon) if sm.lon is not None else "",
            ] + [sm.extra.get(k, "") for k in all_extra_keys]
            f.write("\t".join(row) + "\n")

    print(f"Wrote metadata.tsv → {out_path}  ({len(results)} samples, {len(header)} columns)")


def print_summary(results: list[SampleMeta]):
    """Print a formatted summary table."""
    try:
        from rich.table import Table
        from rich.console import Console
        console = Console()
        table = Table(title="SRA Metadata Summary", show_lines=True)
        table.add_column("Sample",          style="cyan",  no_wrap=True)
        table.add_column("Accession",       style="green", no_wrap=True)
        table.add_column("Location",        style="white")
        table.add_column("Date",            style="yellow", no_wrap=True)
        table.add_column("Isolation",       style="white")
        table.add_column("Coords",          style="dim",   no_wrap=True)
        table.add_column("Study (short)",   style="dim")
        for sm in results:
            coords = f"{sm.lat:.3f}, {sm.lon:.3f}" if sm.lat else "—"
            study_short = sm.study[:45] + "…" if len(sm.study) > 45 else sm.study
            table.add_row(
                sm.sample_name, sm.accession,
                sm.geo_loc_name or "—",
                sm.collection_date or "—",
                sm.isolation_source or "—",
                coords, study_short or "—",
            )
        console.print(table)
    except ImportError:
        # Fallback plain table
        print("\n" + "─" * 110)
        print(f"{'Sample':<40} {'Accession':<14} {'Location':<30} {'Date':<14} {'Coords'}")
        print("─" * 110)
        for sm in results:
            coords = f"{sm.lat:.3f},{sm.lon:.3f}" if sm.lat else "—"
            print(f"{sm.sample_name:<40} {sm.accession:<14} {sm.geo_loc_name[:28]:<30} "
                  f"{sm.collection_date:<14} {coords}")
        print("─" * 110)


if __name__ == "__main__":
    input_dir = Path(INPUT_DIR)
    if not input_dir.exists():
        sys.exit(f"ERROR: INPUT_DIR does not exist: {INPUT_DIR}")

    results = fetch_all(input_dir)

    if not results:
        sys.exit(0)

    print_summary(results)
    write_site_locations(results, SITE_LOCATIONS_JSON)
    write_metadata_tsv(results, METADATA_TSV)

    print(f"\nDone. {len(results)} sample(s) processed.")
    print(f"  Map pins:     {SITE_LOCATIONS_JSON}")
    print(f"  Metadata TSV: {METADATA_TSV}")
    print("\nTip: the metadata.tsv can be used to label samples for differential")
    print("abundance analysis (e.g. group by geo_loc_name or collection_date).")
