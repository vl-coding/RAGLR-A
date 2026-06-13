"""
update_arxiv_data.py

Purpose:
    Download or update arXiv metadata for the RAG Literature Review Assistant.

What this script does:
    1. Harvests arXiv metadata using OAI-PMH.
    2. Supports initial full harvest.
    3. Supports incremental updates using saved harvest state.
    4. Parses title, abstract, authors, categories, dates, and URLs.
    5. Saves clean records to data/processed/arxiv_papers.jsonl.
    6. Deduplicates papers by arXiv ID.
    7. Writes update_state.json and manifest.json.

Why OAI-PMH:
    arXiv recommends OAI-PMH for bulk metadata harvesting and keeping
    an up-to-date copy of arXiv metadata.

Metadata format:
    Uses metadataPrefix=arXiv (configs/config.yaml: data.oai_metadata_prefix),
    which reliably populates <categories> (including cross-listings) and
    structured authors/dates -- unlike oai_dc, where dc:subject rarely
    contains category codes.

Example commands:

    First small test:
        python scripts/update_arxiv_data.py --max-records 1000

    Full initial harvest:
        python scripts/update_arxiv_data.py

    Incremental update:
        python scripts/update_arxiv_data.py --incremental

    Rebuild from scratch:
        python scripts/update_arxiv_data.py --reset
"""

import argparse
import json
import time
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import requests
import yaml


# -----------------------------
# Basic utilities
# -----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def normalize_whitespace(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


# Inline LaTeX formatting commands whose argument should be kept verbatim,
# e.g. "\textbf{robust}" -> "robust".
_LATEX_TEXT_COMMAND_RE = re.compile(
    r"\\(?:textbf|textit|textsc|texttt|emph|mathrm|mathbf|mathit|mathcal)\{([^{}]*)\}"
)

# Escaped special characters that LaTeX renders as the literal character,
# e.g. "100\%" -> "100%".
_LATEX_ESCAPED_CHAR_RE = re.compile(r"\\([%&_#\$])")


def clean_latex_artifacts(text: Optional[str]) -> str:
    """Strip a conservative set of unrendered LaTeX artifacts from OAI-PMH text.

    Some arXiv abstracts/titles include raw LaTeX source rather than rendered
    text. This handles the common, low-risk cases: inline formatting commands
    (unwrapped to their argument), escaped special characters (unescaped to
    the literal character), "\\" line breaks, and bare "$...$" math
    delimiters (the delimiters are dropped; their contents are left as-is
    since BM25 already ignores non-alphanumeric symbols).
    """
    if not text:
        return ""

    text = _LATEX_TEXT_COMMAND_RE.sub(r"\1", text)
    text = _LATEX_ESCAPED_CHAR_RE.sub(r"\1", text)
    text = text.replace("\\\\", " ")
    text = text.replace("$", "")
    return text


def local_name(tag: str) -> str:
    """
    Converts '{namespace}title' into 'title'.
    """
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def get_child_text_by_local_name(element: ET.Element, name: str) -> Optional[str]:
    for child in element.iter():
        if local_name(child.tag) == name:
            return child.text
    return None


# -----------------------------
# State management
# -----------------------------

def load_state(path: str) -> dict:
    if not Path(path).exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str, state: dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_existing_papers(path: str) -> Dict[str, dict]:
    """
    Loads existing processed paper records into a dictionary:
        arxiv_id -> paper dict
    """
    papers = {}

    if not Path(path).exists():
        return papers

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                paper = json.loads(line)
                arxiv_id = paper.get("arxiv_id")

                if arxiv_id:
                    papers[arxiv_id] = paper

            except json.JSONDecodeError:
                continue

    return papers


def save_papers_jsonl(papers_by_id: Dict[str, dict], path: str) -> None:
    """
    Saves papers sorted by arXiv ID for reproducibility.
    """
    ensure_parent(path)

    with open(path, "w", encoding="utf-8") as f:
        for arxiv_id in sorted(papers_by_id.keys()):
            f.write(json.dumps(papers_by_id[arxiv_id], ensure_ascii=False) + "\n")


# -----------------------------
# Harvest checkpointing
#
# A long harvest can take hours and span thousands of OAI requests, so a
# single transient network failure shouldn't discard everything harvested
# so far. As batches are harvested, parsed papers and the current
# resumption token are appended/saved to disk; `--resume` picks back up
# from there instead of restarting from batch #1.
# -----------------------------

def load_checkpoint(papers_path: str, state_path: str) -> tuple[List[dict], Optional[str]]:
    papers: List[dict] = []

    if Path(papers_path).exists():
        with open(papers_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    papers.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    resumption_token = None

    if Path(state_path).exists():
        with open(state_path, "r", encoding="utf-8") as f:
            resumption_token = json.load(f).get("resumption_token")

    return papers, resumption_token


def clear_checkpoint(papers_path: str, state_path: str) -> None:
    for path_str in (papers_path, state_path):
        path = Path(path_str)
        if path.exists():
            path.unlink()


# -----------------------------
# OAI-PMH request handling
# -----------------------------

def build_oai_url(base_url: str, params: dict) -> str:
    return base_url + "?" + urlencode(params)


def fetch_oai_xml(
    base_url: str,
    params: dict,
    max_retries: int = 5,
    sleep_seconds: float = 3.0,
    timeout_seconds: float = 120.0,
) -> ET.Element:
    """
    Makes a polite OAI-PMH request and parses XML.

    If rate-limited, temporarily unavailable, or affected by a transient
    network error (connection drop, read timeout), retries with backoff.
    """
    url = build_oai_url(base_url, params)

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout_seconds)
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"OAI request failed after {max_retries} attempts: {exc}"
                ) from exc

            wait = sleep_seconds * attempt
            print(f"Request error ({exc.__class__.__name__}: {exc}). Retrying in {wait:.1f}s...")
            time.sleep(wait)
            continue

        if response.status_code == 200:
            return ET.fromstring(response.content)

        if response.status_code in {429, 500, 502, 503, 504}:
            wait = sleep_seconds * attempt
            print(f"Request failed with {response.status_code}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
            continue

        raise RuntimeError(
            f"OAI request failed with status {response.status_code}: {response.text[:500]}"
        )

    raise RuntimeError(f"OAI request failed after {max_retries} attempts.")


def extract_resumption_token(root: ET.Element) -> Optional[str]:
    for element in root.iter():
        if local_name(element.tag) == "resumptionToken":
            token = normalize_whitespace(element.text)
            return token if token else None

    return None


# -----------------------------
# Metadata parsing
# -----------------------------

def extract_year(date_values: List[str]) -> Optional[int]:
    """
    Attempts to extract a year from OAI dc:date values.
    """
    for value in date_values:
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            return int(match.group(0))

    return None


def parse_authors_arxiv(record: ET.Element) -> List[str]:
    """
    Extracts authors from the arXiv metadata format's
    <authors><author><keyname>/<forenames></author></authors> elements,
    formatted as "Keyname, Forenames" to match the prior dc:creator
    convention.
    """
    authors = []

    for element in record.iter():
        if local_name(element.tag) != "author":
            continue

        keyname = None
        forenames = None

        for child in element:
            name = local_name(child.tag)

            if name == "keyname" and child.text:
                keyname = normalize_whitespace(child.text)
            elif name == "forenames" and child.text:
                forenames = normalize_whitespace(child.text)

        if keyname and forenames:
            authors.append(f"{keyname}, {forenames}")
        elif keyname:
            authors.append(keyname)

    return authors


def parse_oai_record(record: ET.Element, drop_stats: Optional[Dict[str, int]] = None) -> Optional[dict]:
    """
    Parses one OAI-PMH record using the "arXiv" metadata format
    (metadataPrefix=arXiv).

    Expected useful fields (within the <arXiv> metadata element):
        id
        title
        abstract
        authors/author/keyname, forenames
        categories (space-separated, e.g. "cs.LG cs.AI stat.ML")
        created, updated

    Unlike the oai_dc format, <categories> is reliably populated here,
    including cross-listings, with the first entry being the primary
    category.

    The parser uses local XML tag names so it is namespace-tolerant.

    If drop_stats is provided, increments a counter under the reason a
    record was dropped ("deleted", "no_arxiv_id", "missing_title_or_abstract")
    or kept ("kept").
    """
    def _bump(reason: str) -> None:
        if drop_stats is not None:
            drop_stats[reason] = drop_stats.get(reason, 0) + 1

    # Skip deleted records
    for header in record:
        if local_name(header.tag) == "header":
            if header.attrib.get("status") == "deleted":
                _bump("deleted")
                return None

    arxiv_id = get_child_text_by_local_name(record, "id")

    if not arxiv_id:
        _bump("no_arxiv_id")
        return None

    arxiv_id = arxiv_id.strip()

    title_text = get_child_text_by_local_name(record, "title")
    title = normalize_whitespace(clean_latex_artifacts(title_text)) if title_text else ""

    abstract_text = get_child_text_by_local_name(record, "abstract")
    abstract = normalize_whitespace(clean_latex_artifacts(abstract_text)) if abstract_text else ""

    authors = parse_authors_arxiv(record)

    categories_text = get_child_text_by_local_name(record, "categories")
    categories = categories_text.split() if categories_text else []
    primary_category = categories[0] if categories else None

    created = get_child_text_by_local_name(record, "created")
    updated = get_child_text_by_local_name(record, "updated")

    created = normalize_whitespace(created) if created else None
    updated = normalize_whitespace(updated) if updated else None

    year = extract_year([created]) if created else None

    if year is None:
        # Fallback: arXiv IDs after 2007 often start with YYMM.
        # Example: 2301.12345 -> 2023.
        match = re.match(r"^(\d{2})(\d{2})\.", arxiv_id)
        if match:
            yy = int(match.group(1))
            year = 2000 + yy if yy < 90 else 1900 + yy

    if year is None:
        year = 0

    url = f"https://arxiv.org/abs/{arxiv_id}"

    if not title or not abstract:
        # Keep the system clean by skipping records without enough retrieval text.
        _bump("missing_title_or_abstract")
        return None

    _bump("kept")

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "primary_category": primary_category,
        "categories": categories,
        "category_metadata": [],
        "year": year,
        "published_date": created,
        "updated_date": updated,
        "url": url,
        "source": "arxiv_oai_pmh",
        "updated_at_utc": utc_now_iso(),
    }


def records_from_oai_root(root: ET.Element) -> List[ET.Element]:
    return [element for element in root.iter() if local_name(element.tag) == "record"]


# -----------------------------
# Harvest logic
# -----------------------------

def harvest_oai_records(
    base_url: str,
    metadata_prefix: str,
    from_date: Optional[str] = None,
    max_records: Optional[int] = None,
    sleep_seconds: float = 3.0,
    drop_stats: Optional[Dict[str, int]] = None,
    harvested: Optional[List[dict]] = None,
    resumption_token: Optional[str] = None,
    checkpoint_papers_path: Optional[str] = None,
    checkpoint_state_path: Optional[str] = None,
) -> List[dict]:
    """
    Harvests records from arXiv OAI-PMH.

    If from_date is None:
        performs broad harvest.

    If from_date is provided:
        performs incremental harvest from that date.

    If drop_stats is provided, accumulates per-record outcome counts
    (see parse_oai_record) across the whole harvest.

    `harvested` and `resumption_token` allow resuming a previously
    interrupted harvest (see load_checkpoint). When `checkpoint_papers_path`
    and `checkpoint_state_path` are given, newly parsed papers and the
    current resumption token are persisted after every batch.

    Note:
        For a complete production harvest, expect many requests.
        Use --max-records while testing.
    """
    harvested = list(harvested) if harvested else []
    request_count = 0

    checkpoint_file = None
    if checkpoint_papers_path:
        ensure_parent(checkpoint_papers_path)
        checkpoint_file = open(checkpoint_papers_path, "a", encoding="utf-8")

    try:
        while True:
            if resumption_token:
                params = {
                    "verb": "ListRecords",
                    "resumptionToken": resumption_token,
                }
            else:
                params = {
                    "verb": "ListRecords",
                    "metadataPrefix": metadata_prefix,
                }

                if from_date:
                    params["from"] = from_date

            request_count += 1
            print(f"Fetching OAI batch #{request_count}...")

            root = fetch_oai_xml(base_url=base_url, params=params)

            records = records_from_oai_root(root)

            for record in records:
                paper = parse_oai_record(record, drop_stats=drop_stats)

                if paper:
                    harvested.append(paper)

                    if checkpoint_file:
                        checkpoint_file.write(json.dumps(paper, ensure_ascii=False) + "\n")

                if max_records and len(harvested) >= max_records:
                    print(f"Reached max_records={max_records}. Stopping test harvest.")
                    return harvested

            resumption_token = extract_resumption_token(root)

            print(f"Harvested valid papers so far: {len(harvested)}")

            if checkpoint_file:
                checkpoint_file.flush()

            if checkpoint_state_path:
                save_state(checkpoint_state_path, {"resumption_token": resumption_token})

            if not resumption_token:
                break

            # Be polite to arXiv.
            time.sleep(sleep_seconds)
    finally:
        if checkpoint_file:
            checkpoint_file.close()

    return harvested


# -----------------------------
# Manifest
# -----------------------------

def write_manifest(
    manifest_path: str,
    papers_by_id: Dict[str, dict],
    update_mode: str,
    new_or_updated_count: int,
    drop_stats: Optional[Dict[str, int]] = None,
) -> None:
    ensure_parent(manifest_path)

    category_counts = {}

    for paper in papers_by_id.values():
        for cat in paper.get("categories", []):
            category_counts[cat] = category_counts.get(cat, 0) + 1

    years = [
        paper.get("year")
        for paper in papers_by_id.values()
        if isinstance(paper.get("year"), int) and paper.get("year") > 0
    ]

    manifest = {
        "dataset_name": "arxiv_full_metadata",
        "created_or_updated_at_utc": utc_now_iso(),
        "update_mode": update_mode,
        "total_papers": len(papers_by_id),
        "new_or_updated_records_this_run": new_or_updated_count,
        "min_year": min(years) if years else None,
        "max_year": max(years) if years else None,
        "num_categories": len(category_counts),
        "category_counts": dict(sorted(category_counts.items())),
        "ingestion_drop_stats_this_run": drop_stats or {},
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to project config YAML."
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Harvest only records updated since the last successful harvest."
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore existing processed data and rebuild from harvested records."
    )

    parser.add_argument(
        "--from-date",
        default=None,
        help="Optional OAI-PMH from date in YYYY-MM-DD format. Overrides saved state."
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Testing only: stop after this many valid paper records."
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=3.0,
        help="Seconds to sleep between OAI-PMH requests."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously interrupted harvest from its checkpoint, "
             "if one exists, instead of starting from batch #1."
    )

    args = parser.parse_args()

    config = load_yaml(args.config)

    processed_path = config["data"]["processed_path"]
    base_url = config["data"]["oai_base_url"]
    metadata_prefix = config["data"].get("oai_metadata_prefix", "oai_dc")
    state_path = config["paths"]["update_state"]
    manifest_path = config["paths"]["manifest"]

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path)

    if args.reset:
        print("Reset mode: existing processed records will be ignored.")
        existing_papers = {}
    else:
        existing_papers = load_existing_papers(processed_path)

    print(f"Existing processed papers loaded: {len(existing_papers)}")

    from_date = None
    update_mode = "full_or_test_harvest"

    if args.from_date:
        from_date = args.from_date
        update_mode = "manual_from_date_incremental"

    elif args.incremental:
        last_harvest_date = state.get("last_successful_harvest_date")

        if last_harvest_date:
            from_date = last_harvest_date
        else:
            # Conservative fallback: fetch updates from yesterday if no state exists.
            from_date = (
                datetime.now(timezone.utc).date() - timedelta(days=1)
            ).isoformat()

        update_mode = "incremental"

    print(f"Update mode: {update_mode}")
    print(f"OAI from date: {from_date}")

    drop_stats: Dict[str, int] = {}

    checkpoint_papers_path = "artifacts/harvest_checkpoint_papers.jsonl"
    checkpoint_state_path = "artifacts/harvest_checkpoint_state.json"

    initial_harvested: List[dict] = []
    initial_resumption_token: Optional[str] = None

    if args.resume and Path(checkpoint_papers_path).exists():
        initial_harvested, initial_resumption_token = load_checkpoint(
            checkpoint_papers_path, checkpoint_state_path
        )
        print(
            f"Resuming from checkpoint: {len(initial_harvested)} papers already "
            f"harvested, resumption_token={'set' if initial_resumption_token else 'none'}"
        )

    # Only checkpoint full/incremental harvests, not bounded test runs.
    use_checkpoint = args.max_records is None

    harvested_papers = harvest_oai_records(
        base_url=base_url,
        metadata_prefix=metadata_prefix,
        from_date=from_date,
        max_records=args.max_records,
        sleep_seconds=args.sleep_seconds,
        drop_stats=drop_stats,
        harvested=initial_harvested,
        resumption_token=initial_resumption_token,
        checkpoint_papers_path=checkpoint_papers_path if use_checkpoint else None,
        checkpoint_state_path=checkpoint_state_path if use_checkpoint else None,
    )

    if use_checkpoint:
        clear_checkpoint(checkpoint_papers_path, checkpoint_state_path)

    print(f"Valid harvested papers this run: {len(harvested_papers)}")
    print(f"Ingestion outcome counts this run: {drop_stats}")

    new_or_updated_count = 0

    for paper in harvested_papers:
        arxiv_id = paper["arxiv_id"]

        if arxiv_id not in existing_papers:
            new_or_updated_count += 1
        else:
            # Count it as updated if the paper dict changed.
            if existing_papers[arxiv_id] != paper:
                new_or_updated_count += 1

        existing_papers[arxiv_id] = paper

    save_papers_jsonl(existing_papers, processed_path)

    current_date = datetime.now(timezone.utc).date().isoformat()

    state = {
        "last_successful_harvest_at_utc": utc_now_iso(),
        "last_successful_harvest_date": current_date,
        "processed_path": processed_path,
        "total_papers": len(existing_papers),
        "last_update_mode": update_mode,
        "last_run_new_or_updated_records": new_or_updated_count,
    }

    save_state(state_path, state)

    write_manifest(
        manifest_path=manifest_path,
        papers_by_id=existing_papers,
        update_mode=update_mode,
        new_or_updated_count=new_or_updated_count,
        drop_stats=drop_stats,
    )

    print("Update complete.")
    print(f"Total processed papers: {len(existing_papers)}")
    print(f"New or updated records this run: {new_or_updated_count}")
    print(f"Saved processed data to: {processed_path}")
    print(f"Saved state to: {state_path}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()