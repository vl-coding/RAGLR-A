# Data Card — RAGLR-A

## Source

**Provider:** arXiv.org  
**Protocol:** OAI-PMH (Open Archives Initiative Protocol for Metadata Harvesting)  
**Endpoint:** `https://oaipmh.arxiv.org/oai`  
**Metadata format:** Dublin Core (`oai_dc`)  
**License:** arXiv metadata is made freely available under the [arXiv non-exclusive license](https://arxiv.org/help/license). Paper full-texts are not harvested.

---

## What is collected

Each record corresponds to one arXiv preprint. The following fields are extracted from OAI-PMH Dublin Core metadata:

| Field | Source element | Notes |
|---|---|---|
| `arxiv_id` | `dc:identifier` | Extracted from `oai:arXiv.org:<id>` or abs URL |
| `title` | `dc:title` | Whitespace-normalized |
| `abstract` | `dc:description` | First `dc:description` value |
| `authors` | `dc:creator` | All creator values |
| `categories` | `dc:subject` | Parsed via arXiv category regex; **empty for ~99.99% of records** — arXiv's `oai_dc` feed rarely populates `dc:subject` with category codes. See [LIMITATIONS.md](LIMITATIONS.md). |
| `year` | `dc:date` | First 4-digit year found; falls back to arXiv ID prefix |
| `url` | `dc:identifier` | `https://arxiv.org/abs/<id>` |

Full-text PDFs, LaTeX sources, reference lists, and citation counts are **not** collected.

---

## What is excluded

Records are dropped during ingestion if:
- The arXiv ID cannot be parsed from any identifier field
- Both `title` and `abstract` are empty after whitespace normalization
- The OAI-PMH record has `status="deleted"`

---

## Processed format

Papers are stored as newline-delimited JSON (`data/processed/arxiv_papers.jsonl`). Each line is a valid JSON object matching the `Paper` Pydantic schema:

```json
{
  "arxiv_id": "2301.07041",
  "title": "Example Paper Title",
  "abstract": "We present a method for ...",
  "authors": ["Author One", "Author Two"],
  "categories": [],
  "year": 2023,
  "url": "https://arxiv.org/abs/2301.07041"
}
```

In practice `categories` is `[]` for nearly all records (see note above) — shown here for schema completeness, not as a representative example.

Records are sorted by `arxiv_id` for reproducibility. Deduplication is by `arxiv_id` — the most recently harvested version of a paper overwrites earlier versions.

---

## Taxonomy coverage

The harvested corpus spans all 14 top-level arXiv field groups:

| Field | Example categories |
|---|---|
| Computer Science | cs.AI, cs.LG, cs.CL, cs.CV, cs.RO |
| Mathematics | math.CO, math.ST, math.OC |
| Physics | physics.comp-ph, physics.flu-dyn |
| Astrophysics | astro-ph.CO, astro-ph.GA |
| Condensed Matter | cond-mat.str-el, cond-mat.supr-con |
| High Energy Physics | hep-th, hep-ph, hep-ex |
| Nuclear Physics | nucl-ex, nucl-th |
| Nonlinear Sciences | nlin.CD, nlin.SI |
| Statistics | stat.ML, stat.TH |
| Quantitative Biology | q-bio.NC, q-bio.GN |
| Quantitative Finance | q-fin.PM, q-fin.RM |
| Economics | econ.EM, econ.TH |
| Electrical Engineering & Systems Science | eess.SP, eess.IV |
| Other Physics | gr-qc, quant-ph, math-ph |

This table describes arXiv's taxonomy as a reference for the corpus's topical scope, not the populated `categories` field — which, as noted above, is empty for nearly all harvested records. Category codes, when present, come from author self-reporting and reflect arXiv's taxonomy at harvest time. A paper may appear in multiple field groups if it is cross-listed.

---

## Update cadence

| Mode | Command | Description |
|---|---|---|
| Initial harvest | `python scripts/update_arxiv_data.py` | Full OAI-PMH harvest from all time |
| Incremental update | `python scripts/update_arxiv_data.py --incremental` | Fetches records updated since last run |
| Test harvest | `python scripts/update_arxiv_data.py --max-records 1000` | Stops after 1,000 valid records |
| Rebuild from scratch | `python scripts/update_arxiv_data.py --reset` | Ignores existing processed data |

State is saved to `artifacts/update_state.json` after each successful harvest, including the last harvest date used for incremental updates.

---

## Manifest

After each harvest or index build, `artifacts/manifest.json` records:

- Total paper count
- Harvest timestamp
- Category distribution (counts per arXiv category code) — currently near-zero across the board, since `categories` is empty for ~99.99% of records
- Year range of the corpus
- Update mode used

---

## Intended use

This dataset is intended solely for **academic research and literature discovery**. The paper metadata (titles, abstracts, authors, categories) is used to build retrieval indexes and enable semantic search. No full-text content is stored or redistributed.

---

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for a detailed discussion of data quality issues, coverage gaps, and known parsing failures.
