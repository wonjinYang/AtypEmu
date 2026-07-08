"""Source catalog constants and benchmark seed metadata for AtypEmu."""

from __future__ import annotations

import hashlib
from typing import Any


ALLOWED_STATE_CLASSES = {
    "free_idp_ensemble",
    "saxs_validation",
    "bound_disorder_complex",
    "protean_annotation",
    "fuzzy_complex",
    "folded_nmr_control",
}

ALLOWED_USE_TIERS = {
    "training_candidate",
    "validation_only",
    "benchmark_only",
    "metadata_only",
}

SOURCE_CATALOG = {
    "PED": {
        "tier": 1,
        "collection_order": 1,
        "collection_role": "direct ensemble source",
        "default_use_tiers": ["training_candidate", "benchmark_only"],
        "default_state_classes": ["free_idp_ensemble"],
        "official_urls": {
            "home": "https://proteinensemble.org",
            "api": "https://proteinensemble.org/api",
            "paper": ("https://academic.oup.com/nar/article/52/D1/D536/7334090"),
        },
        "notes": (
            "Primary external coordinate-ensemble source for free-state IDP "
            "ensembles and benchmark candidates."
        ),
    },
    "MobiDB": {
        "tier": 2,
        "collection_order": 2,
        "collection_role": "annotation and discovery source",
        "default_use_tiers": ["metadata_only"],
        "default_state_classes": ["protean_annotation"],
        "official_urls": {
            "home": "https://mobidb.org",
            "download": "https://mobidb.org/api/download_page",
            "paper": ("https://academic.oup.com/nar/article/53/D1/D495/7848843"),
        },
        "notes": (
            "Annotation-first source for disorder, mobility, and construct "
            "selection metadata."
        ),
    },
    "SASBDB": {
        "tier": 2,
        "collection_order": 3,
        "collection_role": "experimental validation source",
        "default_use_tiers": ["validation_only"],
        "default_state_classes": ["saxs_validation"],
        "official_urls": {
            "home": "https://www.sasbdb.org/",
            "swagger": "https://www.sasbdb.org/rest-api/swagger/swagger.json",
            "api_docs": "https://www.sasbdb.org/rest-api/docs/#/summary/list_path",
            "registry": "https://www.re3data.org/repository/r3d100012273",
        },
        "notes": (
            "Solution scattering source for SAXS/SANS validation and profile "
            "cross-links, not a general-purpose ensemble-coordinate corpus."
        ),
    },
    "IDEAL": {
        "tier": 3,
        "collection_order": 4,
        "collection_role": "annotation and benchmark source",
        "default_use_tiers": ["benchmark_only", "metadata_only"],
        "default_state_classes": ["protean_annotation"],
        "official_urls": {
            "home": "https://www.ideal-db.org/",
            "download": "https://www.ideal-db.org/current.html",
        },
        "notes": (
            "Curated IDP and ProS annotations used for construct boundaries "
            "and disorder-to-order flags."
        ),
    },
    "DIBS": {
        "tier": 3,
        "collection_order": 5,
        "collection_role": "bound-complex benchmark source",
        "default_use_tiers": ["benchmark_only", "metadata_only"],
        "default_state_classes": ["bound_disorder_complex"],
        "official_urls": {
            "downloads": "https://dibs.enzim.ttk.mta.hu/downloads.php",
        },
        "notes": (
            "Curated coupled folding and binding complexes. Kept out of the "
            "free-state training pool."
        ),
    },
    "MFIB": {
        "tier": 3,
        "collection_order": 6,
        "collection_role": "bound-complex benchmark source",
        "default_use_tiers": ["benchmark_only", "metadata_only"],
        "default_state_classes": ["bound_disorder_complex"],
        "official_urls": {
            "home": "https://mfib.pbrg.hu/",
            "downloads": "https://mfib.pbrg.hu/downloads.php",
        },
        "notes": (
            "Mutual folding complexes used as bound-state benchmarks, not "
            "free-state training data."
        ),
    },
    "FuzDB": {
        "tier": 3,
        "collection_order": 7,
        "collection_role": "fuzzy-complex benchmark source",
        "default_use_tiers": ["benchmark_only", "metadata_only"],
        "default_state_classes": ["fuzzy_complex"],
        "official_urls": {
            "home": "https://fuzdb.org",
            "browse": "https://fuzdb.org/browse",
            "entries_api": "https://fuzdb.org/api/entries",
            "paper": ("https://academic.oup.com/nar/article/45/D1/D228/2333923"),
            "paper_v4": ("https://academic.oup.com/nar/article/50/D1/D509/6430485"),
        },
        "notes": (
            "Curated fuzzy-complex resource used for benchmark discovery, "
            "cross-linking, and full-prose annotation rather than bulk "
            "coordinate ingestion."
        ),
    },
}

CURATED_SOURCE_ASSETS = {
    "PED": [
        {
            "label": "home",
            "url": "https://proteinensemble.org/",
            "category": "pages",
            "relative_path": "pages/home.html",
        },
        {
            "label": "api_landing",
            "url": "https://proteinensemble.org/api",
            "category": "pages",
            "relative_path": "pages/api.html",
        },
    ],
    "MobiDB": [
        {
            "label": "home",
            "url": "https://mobidb.org/",
            "category": "pages",
            "relative_path": "pages/home.html",
        },
    ],
    "SASBDB": [
        {
            "label": "home",
            "url": "https://www.sasbdb.org/",
            "category": "pages",
            "relative_path": "pages/home.html",
        },
        {
            "label": "help",
            "url": "https://www.sasbdb.org/help/",
            "category": "pages",
            "relative_path": "pages/help.html",
        },
        {
            "label": "api_docs",
            "url": "https://www.sasbdb.org/rest-api/docs/",
            "category": "pages",
            "relative_path": "pages/api_docs.html",
        },
        {
            "label": "swagger",
            "url": "https://www.sasbdb.org/rest-api/swagger/swagger.json",
            "category": "pages",
            "relative_path": "pages/swagger.json",
        },
    ],
    "IDEAL": [
        {
            "label": "current_page",
            "url": "https://www.ideal-db.org/current.html",
            "category": "pages",
            "relative_path": "pages/current.html",
        },
        {
            "label": "ideal_xml",
            "url": "https://www.ideal-db.org/download/current/IDEAL.xml.gz",
            "category": "downloads",
            "relative_path": "downloads/IDEAL.xml.gz",
        },
        {
            "label": "ideal_rdf",
            "url": "https://www.ideal-db.org/download/current/IDEAL.rdf.gz",
            "category": "downloads",
            "relative_path": "downloads/IDEAL.rdf.gz",
        },
    ],
    "DIBS": [
        {
            "label": "downloads_page",
            "url": "https://dibs.pbrg.hu/downloads.php",
            "category": "pages",
            "relative_path": "pages/downloads.html",
        },
        {
            "label": "complete_xml",
            "url": "https://dibs.pbrg.hu/downloads/DIBS_complete.xml",
            "category": "downloads",
            "relative_path": "downloads/DIBS_complete.xml",
        },
        {
            "label": "complete_xsd",
            "url": "https://dibs.pbrg.hu/downloads/DIBS_complete.xsd",
            "category": "downloads",
            "relative_path": "downloads/DIBS_complete.xsd",
        },
        {
            "label": "complete_txt",
            "url": "https://dibs.pbrg.hu/downloads/DIBS_complete.txt",
            "category": "downloads",
            "relative_path": "downloads/DIBS_complete.txt",
        },
        {
            "label": "format_definition",
            "url": "https://dibs.pbrg.hu/downloads/DIBS_format_definition.txt",
            "category": "downloads",
            "relative_path": "downloads/DIBS_format_definition.txt",
        },
        {
            "label": "original_structures_zip",
            "url": "https://dibs.pbrg.hu/downloads/DIBS_all_original_PDB_structures.zip",
            "category": "downloads",
            "relative_path": "downloads/DIBS_all_original_PDB_structures.zip",
            "large_asset": True,
        },
        {
            "label": "modified_structures_zip",
            "url": "https://dibs.pbrg.hu/downloads/DIBS_all_modified_PDB_structures.zip",
            "category": "downloads",
            "relative_path": "downloads/DIBS_all_modified_PDB_structures.zip",
            "large_asset": True,
        },
    ],
    "MFIB": [
        {
            "label": "downloads_page",
            "url": "https://mfib.pbrg.hu/downloads.php",
            "category": "pages",
            "relative_path": "pages/downloads.html",
        },
        {
            "label": "complete_xml_zip",
            "url": "https://mfib.pbrg.hu/downloads/MFIB_complete_xml.zip",
            "category": "downloads",
            "relative_path": "downloads/MFIB_complete_xml.zip",
        },
        {
            "label": "complete_json_zip",
            "url": "https://mfib.pbrg.hu/downloads/MFIB_complete_json.zip",
            "category": "downloads",
            "relative_path": "downloads/MFIB_complete_json.zip",
        },
        {
            "label": "individual_json",
            "url": "https://mfib.pbrg.hu/downloads/MFIB_individual.json",
            "category": "downloads",
            "relative_path": "downloads/MFIB_individual.json",
        },
        {
            "label": "individual_xsd",
            "url": "https://mfib.pbrg.hu/downloads/MFIB_individual.xsd",
            "category": "downloads",
            "relative_path": "downloads/MFIB_individual.xsd",
        },
        {
            "label": "complete_txt",
            "url": "https://mfib.pbrg.hu/downloads/MFIB_complete.txt",
            "category": "downloads",
            "relative_path": "downloads/MFIB_complete.txt",
        },
        {
            "label": "format_definition",
            "url": "https://mfib.pbrg.hu/downloads/MFIB_format_definition.txt",
            "category": "downloads",
            "relative_path": "downloads/MFIB_format_definition.txt",
        },
        {
            "label": "structure_archive_zip",
            "url": "https://mfib.pbrg.hu/downloads/MFIB_download_pdb_cif_ALL.zip",
            "category": "downloads",
            "relative_path": "downloads/MFIB_download_pdb_cif_ALL.zip",
            "large_asset": True,
        },
    ],
    "FuzDB": [
        {
            "label": "home",
            "url": "https://fuzdb.org/",
            "category": "pages",
            "relative_path": "pages/home.html",
        },
        {
            "label": "browse",
            "url": "https://fuzdb.org/browse",
            "category": "pages",
            "relative_path": "pages/browse.html",
        },
        {
            "label": "entries_api",
            "url": "https://fuzdb.org/api/entries",
            "category": "downloads",
            "relative_path": "downloads/entries.json",
        },
    ],
}


def _make_record(
    source_db: str,
    external_id: str,
    protein_name: str,
    state_class: str,
    use_tier: str,
    benchmark_track: str,
    *,
    assets: dict[str, list[dict[str, str]]] | None = None,
    cross_refs: dict[str, Any] | None = None,
    construct_start: int | None = None,
    construct_end: int | None = None,
    sequence: str | None = None,
    uniprot_accession: str | None = None,
) -> dict[str, Any]:
    """Create one normalized external benchmark record.

    Args:
        source_db: Database or provenance label.
        external_id: Stable external identifier.
        protein_name: Human-readable construct or protein name.
        state_class: Normalized structural-state label.
        use_tier: Default collection or modeling role.
        benchmark_track: AtypEmu benchmark family.
        assets: Optional grouped asset payload.
        cross_refs: Optional cross-reference payload.
        construct_start: Optional construct start residue.
        construct_end: Optional construct end residue.
        sequence: Optional explicit sequence for hashing.
        uniprot_accession: Optional UniProt accession.

    Returns:
        JSON-serializable normalized record.
    """
    if state_class not in ALLOWED_STATE_CLASSES:
        raise ValueError(f"Unsupported state_class: {state_class}")
    if use_tier not in ALLOWED_USE_TIERS:
        raise ValueError(f"Unsupported use_tier: {use_tier}")

    sequence_hash = (
        hashlib.sha256(sequence.encode("utf-8")).hexdigest() if sequence else None
    )
    payload = {
        "source_db": source_db,
        "external_id": external_id,
        "protein_name": protein_name,
        "uniprot_accession": uniprot_accession,
        "sequence_hash": sequence_hash,
        "construct_start": construct_start,
        "construct_end": construct_end,
        "state_class": state_class,
        "use_tier": use_tier,
        "benchmark_track": benchmark_track,
        "assets": assets
        or {
            "coordinates": [],
            "profiles": [],
            "annotations": [],
            "publications": [],
        },
        "cross_refs": cross_refs
        or {
            "PED": None,
            "BMRB": None,
            "PDB": None,
            "SASBDB": None,
            "UniProt": None,
            "PubMed": [],
        },
    }
    return payload


BENCHMARK_SEEDS = [
    _make_record(
        source_db="PED",
        external_id="PED1AAD",
        protein_name="beta-synuclein",
        state_class="free_idp_ensemble",
        use_tier="benchmark_only",
        benchmark_track="free_state_idp",
        assets={
            "coordinates": [],
            "profiles": [],
            "annotations": [
                {
                    "label": "PED-derived Data in Brief dataset",
                    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7078294/",
                }
            ],
            "publications": [
                {
                    "label": "PED-derived Data in Brief dataset",
                    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7078294/",
                }
            ],
        },
        cross_refs={
            "PED": "PED1AAD",
            "BMRB": None,
            "PDB": None,
            "SASBDB": None,
            "UniProt": None,
            "PubMed": ["32195305"],
        },
    ),
    _make_record(
        source_db="PED",
        external_id="PED4AAB",
        protein_name="Sendai virus phosphoprotein",
        state_class="free_idp_ensemble",
        use_tier="benchmark_only",
        benchmark_track="free_state_idp",
        assets={
            "coordinates": [],
            "profiles": [],
            "annotations": [
                {
                    "label": "PED-derived Data in Brief dataset",
                    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7078294/",
                }
            ],
            "publications": [
                {
                    "label": "PED-derived Data in Brief dataset",
                    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7078294/",
                }
            ],
        },
        cross_refs={
            "PED": "PED4AAB",
            "BMRB": None,
            "PDB": None,
            "SASBDB": None,
            "UniProt": None,
            "PubMed": ["32195305"],
        },
    ),
    _make_record(
        source_db="PED",
        external_id="PED9AAA",
        protein_name="Sic1/Cdc4",
        state_class="free_idp_ensemble",
        use_tier="benchmark_only",
        benchmark_track="free_state_idp",
        assets={
            "coordinates": [],
            "profiles": [],
            "annotations": [],
            "publications": [
                {
                    "label": "PED review table entry",
                    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4525029/",
                }
            ],
        },
        cross_refs={
            "PED": "PED9AAA",
            "BMRB": None,
            "PDB": None,
            "SASBDB": None,
            "UniProt": None,
            "PubMed": [],
        },
    ),
    _make_record(
        source_db="PED",
        external_id="PED6AAA",
        protein_name="p15 PAF",
        state_class="free_idp_ensemble",
        use_tier="benchmark_only",
        benchmark_track="free_state_idp",
        assets={
            "coordinates": [],
            "profiles": [],
            "annotations": [],
            "publications": [
                {
                    "label": "PED review table entry",
                    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4525029/",
                }
            ],
        },
        cross_refs={
            "PED": "PED6AAA",
            "BMRB": None,
            "PDB": None,
            "SASBDB": None,
            "UniProt": None,
            "PubMed": [],
        },
    ),
    _make_record(
        source_db="PED",
        external_id="PED5AAB",
        protein_name="MKK7",
        state_class="free_idp_ensemble",
        use_tier="benchmark_only",
        benchmark_track="free_state_idp",
        assets={
            "coordinates": [],
            "profiles": [],
            "annotations": [],
            "publications": [
                {
                    "label": "PED review table entry",
                    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4525029/",
                }
            ],
        },
        cross_refs={
            "PED": "PED5AAB",
            "BMRB": None,
            "PDB": None,
            "SASBDB": None,
            "UniProt": None,
            "PubMed": [],
        },
    ),
    _make_record(
        source_db="PED",
        external_id="PED2AAA",
        protein_name="p27 KID",
        state_class="free_idp_ensemble",
        use_tier="benchmark_only",
        benchmark_track="free_state_idp",
        assets={
            "coordinates": [],
            "profiles": [],
            "annotations": [],
            "publications": [
                {
                    "label": "PED review table entry",
                    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4525029/",
                }
            ],
        },
        cross_refs={
            "PED": "PED2AAA",
            "BMRB": None,
            "PDB": None,
            "SASBDB": None,
            "UniProt": None,
            "PubMed": [],
        },
    ),
    _make_record(
        source_db="PED",
        external_id="PED9AAC",
        protein_name="alpha-synuclein",
        state_class="free_idp_ensemble",
        use_tier="benchmark_only",
        benchmark_track="free_state_idp",
        assets={
            "coordinates": [],
            "profiles": [],
            "annotations": [],
            "publications": [
                {
                    "label": "Biophysical Journal reuse example",
                    "url": "https://www.sciencedirect.com/science/article/pii/S0006349518300651",
                }
            ],
        },
        cross_refs={
            "PED": "PED9AAC",
            "BMRB": None,
            "PDB": None,
            "SASBDB": None,
            "UniProt": None,
            "PubMed": [],
        },
    ),
    _make_record(
        source_db="LITERATURE",
        external_id="GB3_CONTROL",
        protein_name="GB3",
        state_class="folded_nmr_control",
        use_tier="benchmark_only",
        benchmark_track="folded_nmr_control",
        assets={
            "coordinates": [],
            "profiles": [],
            "annotations": [],
            "publications": [
                {
                    "label": "GB3 Data in Brief dataset",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/26504890/",
                },
                {
                    "label": "GB3 exact NOE paper",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/26745415/",
                },
            ],
        },
        cross_refs={
            "PED": None,
            "BMRB": None,
            "PDB": ["1IGD"],
            "SASBDB": None,
            "UniProt": None,
            "PubMed": ["26504890", "26745415"],
        },
    ),
]


def _build_source_catalog() -> dict[str, dict[str, Any]]:
    """Return the normalized source-catalog payload."""
    payload: dict[str, dict[str, Any]] = {}
    for source_db, config in SOURCE_CATALOG.items():
        payload[source_db] = {
            "source_db": source_db,
            **config,
            "directory_name": source_db.lower(),
        }
    return payload


def _build_benchmark_registry() -> list[dict[str, Any]]:
    """Return the initial benchmark registry."""
    return [dict(row) for row in BENCHMARK_SEEDS]


def _summarize_benchmark_registry(
    benchmark_registry: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build summary counts for the benchmark registry."""
    by_source: dict[str, int] = {}
    by_track: dict[str, int] = {}
    by_state: dict[str, int] = {}

    for row in benchmark_registry:
        by_source[row["source_db"]] = by_source.get(row["source_db"], 0) + 1
        by_track[row["benchmark_track"]] = by_track.get(row["benchmark_track"], 0) + 1
        by_state[row["state_class"]] = by_state.get(row["state_class"], 0) + 1

    return {
        "records": len(benchmark_registry),
        "source_counts": by_source,
        "track_counts": by_track,
        "state_class_counts": by_state,
    }
