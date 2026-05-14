"""
amr/database.py
Curated genus-level antimicrobial resistance (AMR) profiles for
PathogenIQ alert contextualization.

Sources:
  - WHO Priority Pathogens List 2024 (Critical / High / Medium)
  - CDC AR Threats Report 2019 (Urgent / Serious / Concerning)
  - ESKAPE pathogens (Rice 2008)
  - CARD (Comprehensive Antibiotic Resistance Database)

Each entry defines resistance mechanisms, affected drug classes,
last-resort treatment options, WHO/CDC priority tier, and a
clinical wastewater context note.
"""

# Priority tier constants
CRITICAL = "critical"
HIGH     = "high"
MEDIUM   = "medium"

AMR_DB: dict[str, dict] = {

    # ── ESKAPE / WHO Critical ─────────────────────────────────────────────────

    "Staphylococcus": {
        "who_priority": CRITICAL,
        "cdc_threat": "serious",
        "eskape": True,
        "resistances": [
            {"drug_class": "Beta-lactams",    "mechanism": "MRSA / mecA gene",        "prevalence": "high"},
            {"drug_class": "Glycopeptides",   "mechanism": "VRSA / vanA-vanB",        "prevalence": "rare"},
            {"drug_class": "Macrolides",      "mechanism": "erm genes",               "prevalence": "moderate"},
            {"drug_class": "Fluoroquinolones","mechanism": "gyrA/parC mutations",     "prevalence": "moderate"},
        ],
        "last_resort_drugs": ["Vancomycin", "Linezolid", "Daptomycin", "Ceftaroline"],
        "wbe_note": "MRSA shed in feces; wastewater detection predicts community MRSA burden",
    },

    "Enterococcus": {
        "who_priority": CRITICAL,
        "cdc_threat": "urgent",
        "eskape": True,
        "resistances": [
            {"drug_class": "Glycopeptides",   "mechanism": "VRE / vanA-vanB",         "prevalence": "high"},
            {"drug_class": "Beta-lactams",    "mechanism": "Intrinsic ampicillin-R",   "prevalence": "intrinsic"},
            {"drug_class": "Aminoglycosides", "mechanism": "High-level resistance",    "prevalence": "moderate"},
        ],
        "last_resort_drugs": ["Linezolid", "Daptomycin", "Tigecycline"],
        "wbe_note": "VRE colonization detected in municipal wastewater; indicator of nosocomial transmission",
    },

    "Klebsiella": {
        "who_priority": CRITICAL,
        "cdc_threat": "urgent",
        "eskape": True,
        "resistances": [
            {"drug_class": "Carbapenems",     "mechanism": "KPC / NDM / OXA-48",      "prevalence": "increasing"},
            {"drug_class": "Cephalosporins",  "mechanism": "ESBL (CTX-M, SHV, TEM)",  "prevalence": "high"},
            {"drug_class": "Fluoroquinolones","mechanism": "plasmid-mediated (qnr)",   "prevalence": "moderate"},
            {"drug_class": "Polymyxins",      "mechanism": "mcr-1 plasmid",           "prevalence": "emerging"},
        ],
        "last_resort_drugs": ["Ceftazidime-avibactam", "Meropenem-vaborbactam", "Polymyxin B"],
        "wbe_note": "CRKP (carbapenem-resistant) is a wastewater AMR sentinel; NDM detected globally",
    },

    "Acinetobacter": {
        "who_priority": CRITICAL,
        "cdc_threat": "urgent",
        "eskape": True,
        "resistances": [
            {"drug_class": "Carbapenems",     "mechanism": "OXA-type carbapenemases",  "prevalence": "high"},
            {"drug_class": "Beta-lactams",    "mechanism": "AmpC + TEM + OXA",         "prevalence": "high"},
            {"drug_class": "Polymyxins",      "mechanism": "lpxCAD mutations",         "prevalence": "emerging"},
            {"drug_class": "All major classes","mechanism": "Pandrug resistance (PDR)", "prevalence": "rare"},
        ],
        "last_resort_drugs": ["Polymyxin B", "Colistin", "Sulbactam-durlobactam"],
        "wbe_note": "Hospital wastewater hotspot; environmental persistence on dry surfaces",
    },

    "Pseudomonas": {
        "who_priority": CRITICAL,
        "cdc_threat": "serious",
        "eskape": True,
        "resistances": [
            {"drug_class": "Carbapenems",     "mechanism": "VIM / IMP / GES MBLs",   "prevalence": "increasing"},
            {"drug_class": "Beta-lactams",    "mechanism": "AmpC inducible + OprD loss","prevalence": "high"},
            {"drug_class": "Fluoroquinolones","mechanism": "gyrA/gyrB/nfxB mutations","prevalence": "high"},
            {"drug_class": "Aminoglycosides", "mechanism": "AME enzymes",             "prevalence": "moderate"},
        ],
        "last_resort_drugs": ["Ceftolozane-tazobactam", "Ceftazidime-avibactam", "Imipenem-relebactam"],
        "wbe_note": "Ubiquitous in water; biofilm former; multidrug-resistant strains enriched in sewage",
    },

    "Enterobacter": {
        "who_priority": CRITICAL,
        "cdc_threat": "serious",
        "eskape": True,
        "resistances": [
            {"drug_class": "Cephalosporins",  "mechanism": "AmpC inducible (cephalosporinase)","prevalence": "high"},
            {"drug_class": "Carbapenems",     "mechanism": "KPC / NDM",               "prevalence": "increasing"},
            {"drug_class": "Beta-lactams",    "mechanism": "ESBL (CTX-M)",            "prevalence": "moderate"},
        ],
        "last_resort_drugs": ["Ceftazidime-avibactam", "Meropenem-vaborbactam"],
        "wbe_note": "Third-generation cephalosporin resistance common in clinical isolates; transmitted fecal-oral",
    },

    # ── WHO Critical (non-ESKAPE) ─────────────────────────────────────────────

    "Escherichia": {
        "who_priority": CRITICAL,
        "cdc_threat": "serious",
        "eskape": False,
        "resistances": [
            {"drug_class": "Cephalosporins",  "mechanism": "ESBL (CTX-M-15/-27)",    "prevalence": "very high"},
            {"drug_class": "Carbapenems",     "mechanism": "NDM / OXA-48 / KPC",     "prevalence": "increasing"},
            {"drug_class": "Fluoroquinolones","mechanism": "gyrA mutations + PMQR",   "prevalence": "high"},
            {"drug_class": "Polymyxins",      "mechanism": "mcr-1 plasmid",          "prevalence": "emerging"},
        ],
        "last_resort_drugs": ["Meropenem-vaborbactam", "Aztreonam-avibactam", "Fosfomycin"],
        "wbe_note": "ESBL-E. coli most common AMR organism in wastewater globally; mcr-1 colistin resistance emerging",
    },

    "Salmonella": {
        "who_priority": HIGH,
        "cdc_threat": "serious",
        "eskape": False,
        "resistances": [
            {"drug_class": "Fluoroquinolones","mechanism": "gyrA/parC mutations",     "prevalence": "high"},
            {"drug_class": "Cephalosporins",  "mechanism": "blaCMY-2 / ESBL",        "prevalence": "moderate"},
            {"drug_class": "Multiple classes","mechanism": "MDR ACSSuT phenotype",    "prevalence": "moderate"},
        ],
        "last_resort_drugs": ["Ciprofloxacin", "Azithromycin", "Ceftriaxone"],
        "wbe_note": "DT104 MDR strain shed in livestock wastewater; fluoroquinolone resistance rising",
    },

    "Clostridioides": {
        "who_priority": HIGH,
        "cdc_threat": "urgent",
        "eskape": False,
        "resistances": [
            {"drug_class": "Fluoroquinolones","mechanism": "gyrA/gyrB mutations (BI/NAP1)","prevalence": "high"},
            {"drug_class": "Metronidazole",   "mechanism": "nimB gene (rare)",        "prevalence": "rare"},
        ],
        "last_resort_drugs": ["Vancomycin (oral)", "Fidaxomicin", "Bezlotoxumab"],
        "wbe_note": "Spores survive wastewater treatment; hypervirulent ribotype 027 detected in sewage",
    },

    "Mycobacterium": {
        "who_priority": CRITICAL,
        "cdc_threat": "serious",
        "eskape": False,
        "resistances": [
            {"drug_class": "Rifampicin",      "mechanism": "rpoB mutations (MDR-TB)", "prevalence": "high globally"},
            {"drug_class": "Isoniazid",       "mechanism": "katG/inhA mutations",     "prevalence": "high"},
            {"drug_class": "Fluoroquinolones","mechanism": "gyrA mutations (XDR-TB)", "prevalence": "emerging"},
            {"drug_class": "Bedaquiline",     "mechanism": "atpE mutations",          "prevalence": "rare"},
        ],
        "last_resort_drugs": ["Bedaquiline", "Pretomanid", "Linezolid (BPaL regimen)"],
        "wbe_note": "MDR/XDR-TB DNA detected in municipal wastewater; viable cells in hospital effluent",
    },

    "Neisseria": {
        "who_priority": HIGH,
        "cdc_threat": "urgent",
        "eskape": False,
        "resistances": [
            {"drug_class": "Cephalosporins",  "mechanism": "penA mosaic allele",     "prevalence": "emerging"},
            {"drug_class": "Fluoroquinolones","mechanism": "gyrA/parC mutations",     "prevalence": "very high"},
            {"drug_class": "Macrolides",      "mechanism": "23S rRNA mutations",     "prevalence": "moderate"},
        ],
        "last_resort_drugs": ["Ceftriaxone + Azithromycin", "Gentamicin"],
        "wbe_note": "MDR Neisseria gonorrhoeae detectable in wastewater; correlates with community STI burden",
    },

    "Haemophilus": {
        "who_priority": MEDIUM,
        "cdc_threat": "concerning",
        "eskape": False,
        "resistances": [
            {"drug_class": "Beta-lactams",    "mechanism": "TEM-1 beta-lactamase",   "prevalence": "high"},
            {"drug_class": "Ampicillin",      "mechanism": "BLNAR (non-beta-lactamase)","prevalence": "increasing"},
        ],
        "last_resort_drugs": ["Cefotaxime", "Levofloxacin"],
        "wbe_note": "Ampicillin-resistant strains common; wastewater detection indicates respiratory burden",
    },

    "Campylobacter": {
        "who_priority": HIGH,
        "cdc_threat": "serious",
        "eskape": False,
        "resistances": [
            {"drug_class": "Fluoroquinolones","mechanism": "gyrA C257T mutation",    "prevalence": "very high"},
            {"drug_class": "Macrolides",      "mechanism": "23S rRNA A2075G",        "prevalence": "low"},
            {"drug_class": "Tetracyclines",   "mechanism": "tet(O) gene",            "prevalence": "high"},
        ],
        "last_resort_drugs": ["Azithromycin", "Carbapenem (severe)"],
        "wbe_note": "Leading foodborne pathogen; fluoroquinolone-resistant strains dominate in poultry wastewater",
    },

    "Helicobacter": {
        "who_priority": HIGH,
        "cdc_threat": "concerning",
        "eskape": False,
        "resistances": [
            {"drug_class": "Clarithromycin",  "mechanism": "23S rRNA A2143G",       "prevalence": "high"},
            {"drug_class": "Metronidazole",   "mechanism": "rdxA mutations",         "prevalence": "high"},
            {"drug_class": "Fluoroquinolones","mechanism": "gyrA mutations",         "prevalence": "moderate"},
        ],
        "last_resort_drugs": ["Bismuth quadruple therapy", "Rifabutin-based regimen"],
        "wbe_note": "Clarithromycin resistance >20% in North America; fecal shedding indicates community prevalence",
    },

    "Burkholderia": {
        "who_priority": HIGH,
        "cdc_threat": "serious",
        "eskape": False,
        "resistances": [
            {"drug_class": "All beta-lactams","mechanism": "Intrinsic AmpC + efflux","prevalence": "intrinsic"},
            {"drug_class": "Aminoglycosides", "mechanism": "Intrinsic resistance",   "prevalence": "intrinsic"},
            {"drug_class": "Polymyxins",      "mechanism": "Intrinsic resistance",   "prevalence": "intrinsic"},
        ],
        "last_resort_drugs": ["Meropenem", "TMP-SMX", "Ceftazidime"],
        "wbe_note": "B. pseudomallei (melioidosis): BSL-3; detected in tropical region wastewater",
    },

    "Stenotrophomonas": {
        "who_priority": MEDIUM,
        "cdc_threat": "concerning",
        "eskape": False,
        "resistances": [
            {"drug_class": "Carbapenems",     "mechanism": "Intrinsic L1/L2 MBLs",  "prevalence": "intrinsic"},
            {"drug_class": "Beta-lactams",    "mechanism": "Intrinsic broad resistance","prevalence": "intrinsic"},
            {"drug_class": "Aminoglycosides", "mechanism": "Intrinsic efflux",       "prevalence": "intrinsic"},
        ],
        "last_resort_drugs": ["TMP-SMX", "Ticarcillin-clavulanate", "Minocycline"],
        "wbe_note": "Opportunistic; carbapenem-resistant by nature; enriched in hospital wastewater",
    },

    "Vibrio": {
        "who_priority": HIGH,
        "cdc_threat": "serious",
        "eskape": False,
        "resistances": [
            {"drug_class": "Tetracyclines",   "mechanism": "tet genes on ICEs",     "prevalence": "high (Asia)"},
            {"drug_class": "Fluoroquinolones","mechanism": "gyrA mutations",         "prevalence": "emerging"},
            {"drug_class": "Ampicillin",      "mechanism": "TEM-1",                  "prevalence": "moderate"},
        ],
        "last_resort_drugs": ["Doxycycline", "Azithromycin", "Ciprofloxacin"],
        "wbe_note": "MDR V. cholerae O1 El Tor linked to pandemic waves; wastewater surveillance critical",
    },

    "Shigella": {
        "who_priority": HIGH,
        "cdc_threat": "serious",
        "eskape": False,
        "resistances": [
            {"drug_class": "Fluoroquinolones","mechanism": "gyrA/parC mutations",   "prevalence": "very high"},
            {"drug_class": "Azithromycin",    "mechanism": "mphA / erm genes",      "prevalence": "increasing"},
            {"drug_class": "Multiple",        "mechanism": "XDR-Shigella (MSM outbreaks)","prevalence": "emerging"},
        ],
        "last_resort_drugs": ["Pivmecillinam", "Carbapenem (XDR)"],
        "wbe_note": "XDR Shigella sonnei outbreak strain detectable in wastewater; signals local cluster",
    },

    "Yersinia": {
        "who_priority": CRITICAL,
        "cdc_threat": "urgent",
        "eskape": False,
        "resistances": [
            {"drug_class": "Multiple",        "mechanism": "pFra/pCD1 plasmid-encoded","prevalence": "potential"},
            {"drug_class": "Beta-lactams",    "mechanism": "AmpC + TEM",             "prevalence": "moderate"},
        ],
        "last_resort_drugs": ["Streptomycin", "Gentamicin", "Doxycycline", "Ciprofloxacin"],
        "wbe_note": "Y. pestis: BSL-3/Select Agent; any wastewater detection requires immediate reporting",
    },
}
