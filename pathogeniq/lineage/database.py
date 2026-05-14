"""
lineage/database.py
Species-level context database for PathogenIQ lineage annotation.

Maps Kraken2 species-rank names to known clinically relevant strains,
outbreak lineages, and serotype/variant context. Works with --rank S
Kraken2 output.

Note: True lineage deconvolution (e.g. SARS-CoV-2 variant proportions)
requires read-level alignment tools (Freyja, iVar). This module provides
species-level context — which named strains/serotypes are known to be
particularly virulent, outbreak-associated, or drug-resistant.
"""

# Each entry: species name (as Kraken2 reports it) → context dict
SPECIES_CONTEXT: dict[str, dict] = {

    # ── SARS-CoV-2 ────────────────────────────────────────────────────────────
    "Severe acute respiratory syndrome coronavirus 2": {
        "genus": "Betacoronavirus",
        "common_name": "SARS-CoV-2 / COVID-19",
        "who_label": "pandemic",
        "known_variants": [
            {"name": "Omicron (XBB/JN/KP lineages)", "status": "dominant 2023-2025",
             "notes": "Highly immune evasive; wastewater detection leads clinical cases by ~1 week"},
            {"name": "Delta (B.1.617.2)", "status": "historical",
             "notes": "Higher severity; superseded by Omicron"},
            {"name": "Alpha (B.1.1.7)", "status": "historical", "notes": "First VOC"},
        ],
        "wbe_utility": "Gold standard WBE target; RT-qPCR + Freyja variant deconvolution recommended",
        "risk_elevation": 0.15,
    },

    # ── Salmonella ────────────────────────────────────────────────────────────
    "Salmonella enterica": {
        "genus": "Salmonella",
        "common_name": "Non-typhoidal Salmonella / Typhoid",
        "who_label": "high priority",
        "known_variants": [
            {"name": "S. Typhi (serovar Typhi)", "status": "endemic globally",
             "notes": "Typhoid; XDR Typhi H58 clade expanding in South Asia"},
            {"name": "S. Typhimurium DT104", "status": "MDR outbreak strain",
             "notes": "ACSSuT resistance phenotype; linked to livestock wastewater"},
            {"name": "S. Enteritidis", "status": "leading foodborne",
             "notes": "Poultry/egg reservoir; wastewater detection precedes outbreaks"},
            {"name": "S. Newport", "status": "MDR emerging",
             "notes": "blaCMY-2 mediated AmpC; cattle source"},
        ],
        "wbe_utility": "Species-level confirms fecal contamination origin; serotyping requires culture",
        "risk_elevation": 0.10,
    },

    # ── E. coli ───────────────────────────────────────────────────────────────
    "Escherichia coli": {
        "genus": "Escherichia",
        "common_name": "E. coli (multiple pathotypes)",
        "who_label": "high priority",
        "known_variants": [
            {"name": "O157:H7 (STEC/EHEC)", "status": "outbreak-associated",
             "notes": "Shiga toxin; HUS risk; cattle reservoir; low infectious dose"},
            {"name": "O104:H4 (EAEC-STEC hybrid)", "status": "historical outbreak 2011",
             "notes": "German sprout outbreak, 3,842 cases, 54 deaths"},
            {"name": "ESBL-producing ST131", "status": "dominant clinical clone",
             "notes": "Fluoroquinolone + cephalosporin resistant; global dissemination in wastewater"},
            {"name": "NDM-1 producers", "status": "emerging",
             "notes": "Carbapenem-resistant; detected in municipal wastewater globally"},
        ],
        "wbe_utility": "Indicator organism for fecal contamination; ESBL E. coli prevalence tracks community AMR burden",
        "risk_elevation": 0.05,
    },

    # ── Vibrio ────────────────────────────────────────────────────────────────
    "Vibrio cholerae": {
        "genus": "Vibrio",
        "common_name": "Cholera",
        "who_label": "pandemic pathogen",
        "known_variants": [
            {"name": "O1 El Tor (7th pandemic)", "status": "active pandemic",
             "notes": "Current pandemic strain; MDR variants rising (ICEVchHai1)"},
            {"name": "O139 Bengal", "status": "regional outbreaks",
             "notes": "Non-O1; epidemic potential; South/Southeast Asia"},
        ],
        "wbe_utility": "Direct wastewater detection is a WHO-endorsed cholera early warning signal",
        "risk_elevation": 0.25,
    },

    "Vibrio parahaemolyticus": {
        "genus": "Vibrio",
        "common_name": "Seafood gastroenteritis",
        "who_label": "medium",
        "known_variants": [
            {"name": "O3:K6 pandemic clone", "status": "globally distributed",
             "notes": "Thermostable direct hemolysin (TDH/TRH); seafood-associated"},
        ],
        "wbe_utility": "Indicator of coastal/seafood processing wastewater contamination",
        "risk_elevation": 0.05,
    },

    # ── Yersinia ──────────────────────────────────────────────────────────────
    "Yersinia pestis": {
        "genus": "Yersinia",
        "common_name": "Plague (bubonic / pneumonic / septicemic)",
        "who_label": "select agent / BSL-3",
        "known_variants": [
            {"name": "Biovar Orientalis", "status": "current global strain",
             "notes": "3rd pandemic strain; endemic in rodent reservoirs"},
            {"name": "Biovar Medievalis", "status": "historical",
             "notes": "2nd pandemic (Black Death)"},
        ],
        "wbe_utility": "ANY detection requires immediate biosafety notification. CDC Select Agent.",
        "risk_elevation": 0.50,
    },

    "Yersinia enterocolitica": {
        "genus": "Yersinia",
        "common_name": "Yersiniosis",
        "who_label": "medium",
        "known_variants": [
            {"name": "Bioserotype 4/O:3", "status": "dominant in Europe",
             "notes": "Pork reservoir; cold-tolerant; winter outbreaks"},
            {"name": "Bioserotype 1B/O:8", "status": "North America",
             "notes": "More virulent than European bioserotypes"},
        ],
        "wbe_utility": "Fecal shedding from pigs; wastewater detection near farms indicates risk",
        "risk_elevation": 0.05,
    },

    # ── Klebsiella ────────────────────────────────────────────────────────────
    "Klebsiella pneumoniae": {
        "genus": "Klebsiella",
        "common_name": "Klebsiella pneumonia / UTI / bacteremia",
        "who_label": "critical priority",
        "known_variants": [
            {"name": "Hypervirulent (hvKP) ST23", "status": "emerging",
             "notes": "Liver abscess, endophthalmitis; not yet typically MDR but converging"},
            {"name": "CRKP ST258", "status": "dominant MDR hospital clone",
             "notes": "KPC carbapenemase; responsible for most US CRKP outbreaks"},
            {"name": "NDM-producing strains", "status": "global spread",
             "notes": "Particularly ST11, ST147; detected in municipal wastewater"},
        ],
        "wbe_utility": "CRKP in wastewater indicates hospital effluent mixing; surveillance recommended near ICUs",
        "risk_elevation": 0.12,
    },

    # ── Clostridioides ────────────────────────────────────────────────────────
    "Clostridioides difficile": {
        "genus": "Clostridioides",
        "common_name": "C. diff / CDI",
        "who_label": "urgent",
        "known_variants": [
            {"name": "Ribotype 027 (NAP1/BI)", "status": "hypervirulent",
             "notes": "Binary toxin + 18bp tcdC deletion; fluoroquinolone-resistant; spore former"},
            {"name": "Ribotype 078", "status": "emerging community-associated",
             "notes": "Animal reservoir; community-onset CDI; shared with RT027 virulence factors"},
        ],
        "wbe_utility": "Spores survive conventional wastewater treatment; hospital effluent sentinel",
        "risk_elevation": 0.10,
    },

    # ── Mycobacterium ─────────────────────────────────────────────────────────
    "Mycobacterium tuberculosis": {
        "genus": "Mycobacterium",
        "common_name": "Tuberculosis",
        "who_label": "critical priority",
        "known_variants": [
            {"name": "MDR-TB (RR-TB)", "status": "global epidemic",
             "notes": "Rifampicin-resistant; >450,000 new cases/year globally"},
            {"name": "XDR-TB", "status": "emerging",
             "notes": "Also fluoroquinolone + bedaquiline/linezolid resistant"},
            {"name": "Beijing/East Asian lineage", "status": "dominant in Asia",
             "notes": "Associated with higher transmissibility and drug resistance"},
        ],
        "wbe_utility": "MDR-TB DNA detectable in wastewater; emerging WBE surveillance application",
        "risk_elevation": 0.20,
    },

    # ── Neisseria ─────────────────────────────────────────────────────────────
    "Neisseria gonorrhoeae": {
        "genus": "Neisseria",
        "common_name": "Gonorrhea",
        "who_label": "urgent",
        "known_variants": [
            {"name": "FC428 clone (ceftriaxone-R)", "status": "global spread",
             "notes": "penA-60 allele; reduced ceftriaxone susceptibility; Asia-Pacific origin"},
            {"name": "GISP/GASP strains", "status": "ongoing surveillance",
             "notes": "Declining azithromycin susceptibility globally"},
        ],
        "wbe_utility": "WBE correlates with community gonorrhea burden; useful for drug-resistance trend monitoring",
        "risk_elevation": 0.08,
    },

    # ── Shigella ──────────────────────────────────────────────────────────────
    "Shigella sonnei": {
        "genus": "Shigella",
        "common_name": "Bacillary dysentery (most common in high-income countries)",
        "who_label": "high priority",
        "known_variants": [
            {"name": "XDR MSM outbreak strain", "status": "active global spread",
             "notes": "Fluoroquinolone + azithromycin resistant; sexual transmission network"},
        ],
        "wbe_utility": "Wastewater detection can identify clusters before clinical reporting lags",
        "risk_elevation": 0.08,
    },

    "Shigella flexneri": {
        "genus": "Shigella",
        "common_name": "Bacillary dysentery (developing regions)",
        "who_label": "high priority",
        "known_variants": [
            {"name": "2a global clone", "status": "endemic globally",
             "notes": "Leading serotype in low-income countries; ampicillin + TMP resistant"},
        ],
        "wbe_utility": "Indicator of sanitation breakdown; fecal-oral transmission",
        "risk_elevation": 0.08,
    },

    # ── Acinetobacter ─────────────────────────────────────────────────────────
    "Acinetobacter baumannii": {
        "genus": "Acinetobacter",
        "common_name": "Hospital-acquired pneumonia / wound infection",
        "who_label": "critical priority",
        "known_variants": [
            {"name": "IC1 (OXA-23)", "status": "dominant global clone",
             "notes": "Carbapenem-resistant; hospital wastewater hotspot"},
            {"name": "IC2 (OXA-23/40)", "status": "global spread",
             "notes": "Pan-drug resistant strains emerging"},
        ],
        "wbe_utility": "Hospital effluent indicator; desiccation-tolerant, persists on surfaces",
        "risk_elevation": 0.10,
    },
}
