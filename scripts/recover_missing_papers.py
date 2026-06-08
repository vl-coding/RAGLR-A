"""
recover_missing_papers.py

Recovers the ~383K pre-2007 papers that were lost when arxiv_papers.jsonl
was truncated during an interrupted write on 2026-06-06.

What happened:
    The JSONL is sorted by arxiv_id. New-style IDs (0704.xxxx ... 2601.xxxx)
    sort before old-style IDs (astro-ph/... cs/... etc.) in lexicographic order.
    The truncation happened mid-write at astro-ph/0402445, so all pre-2007
    papers in cond-mat, cs, gr-qc, hep-*, math, nlin, nucl-*, physics, q-*,
    quant-ph, and the later astro-ph papers are missing.

What this script does:
    1. Loads all existing arxiv_ids from the main JSONL into a set (read-only).
    2. Harvests OAI-PMH from 1991-01-01 to 2007-03-31 (covers all old-style IDs).
    3. Writes only NEW papers (not in existing set) to arxiv_recovered.jsonl.

Safe to run while build_dense_index_fast.py is active — it only reads the
main JSONL and writes to a completely separate file.

After dense indexing finishes, run merge_recovered_papers.py to merge and
then re-run build_dense_index_fast.py (idempotent) and build_indexes.py.

Usage:
    python scripts/recover_missing_papers.py
    python scripts/recover_missing_papers.py --sleep-seconds 5
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import requests
import yaml


# OAI-PMH endpoint rejects early start dates, so we do a full undated harvest
# and skip papers already present in the main JSONL.


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_whitespace(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def get_all_child_texts_by_local_name(element: ET.Element, name: str) -> List[str]:
    values = []
    for child in element.iter():
        if local_name(child.tag) == name and child.text:
            values.append(normalize_whitespace(child.text))
    return values


def extract_arxiv_id_from_identifier(identifier: str) -> Optional[str]:
    identifier = identifier.strip()
    if "oai:arXiv.org:" in identifier:
        return identifier.split("oai:arXiv.org:", 1)[1].strip()
    if "arxiv.org/abs/" in identifier:
        return identifier.rsplit("/abs/", 1)[1].strip()
    return None


def extract_year(date_values: List[str]) -> Optional[int]:
    for value in date_values:
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            return int(match.group(0))
    return None


def split_categories(subject_values: List[str]) -> List[str]:
    category_pattern = re.compile(
        r"\b("
        r"astro-ph(?:\.[A-Z]{2})?|"
        r"cond-mat(?:\.[a-z-]+)?|"
        r"cs\.[A-Z]{2}|"
        r"econ\.[A-Z]{2}|"
        r"eess\.[A-Z]{2}|"
        r"gr-qc|"
        r"hep-ex|hep-lat|hep-ph|hep-th|"
        r"math(?:\.[A-Z]{2})?|"
        r"math-ph|"
        r"nlin(?:\.[A-Z]{2})?|"
        r"nucl-ex|nucl-th|"
        r"physics(?:\.[a-z-]+)?|"
        r"q-bio\.[A-Z]{2}|"
        r"q-fin\.[A-Z]{2}|"
        r"quant-ph|"
        r"stat\.[A-Z]{2}"
        r")\b"
    )
    categories = set()
    for subject in subject_values:
        for match in category_pattern.findall(subject):
            categories.add(match)
    return sorted(categories)


def parse_oai_record(record: ET.Element) -> Optional[dict]:
    for header in record:
        if local_name(header.tag) == "header":
            if header.attrib.get("status") == "deleted":
                return None

    title_values = get_all_child_texts_by_local_name(record, "title")
    creator_values = get_all_child_texts_by_local_name(record, "creator")
    description_values = get_all_child_texts_by_local_name(record, "description")
    subject_values = get_all_child_texts_by_local_name(record, "subject")
    identifier_values = get_all_child_texts_by_local_name(record, "identifier")
    date_values = get_all_child_texts_by_local_name(record, "date")

    arxiv_id = None
    url = None
    for identifier in identifier_values:
        possible_id = extract_arxiv_id_from_identifier(identifier)
        if possible_id:
            arxiv_id = possible_id
        if "arxiv.org/abs/" in identifier:
            url = identifier

    if not arxiv_id:
        return None

    title = normalize_whitespace(title_values[0]) if title_values else ""
    abstract = normalize_whitespace(description_values[0]) if description_values else ""

    if not title or not abstract:
        return None

    authors = [normalize_whitespace(a) for a in creator_values if a]
    categories = split_categories(subject_values)
    year = extract_year(date_values)

    if year is None:
        match = re.match(r"^(\d{2})(\d{2})\.", arxiv_id)
        if match:
            yy = int(match.group(1))
            year = 2000 + yy if yy < 90 else 1900 + yy

    if year is None:
        year = 0

    if not url:
        url = f"https://arxiv.org/abs/{arxiv_id}"

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "categories": categories,
        "year": year,
        "url": url,
        "source": "arxiv_oai_pmh",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def fetch_oai_xml(base_url: str, params: dict, max_retries: int = 5, sleep_seconds: float = 3.0) -> ET.Element:
    url = base_url + "?" + urlencode(params)
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                return ET.fromstring(response.content)
            if response.status_code in {429, 500, 502, 503, 504}:
                wait = sleep_seconds * attempt
                print(f"  HTTP {response.status_code}, retrying in {wait:.1f}s...", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"OAI request failed with status {response.status_code}")
        except requests.RequestException as e:
            wait = sleep_seconds * attempt
            print(f"  Request error ({e}), retrying in {wait:.1f}s...", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"OAI request failed after {max_retries} attempts.")


def load_existing_ids(jsonl_path: str) -> set:
    """Read existing arxiv_ids from main JSONL (read-only, safe during dense indexing)."""
    print(f"Loading existing arxiv_ids from {jsonl_path} ...", flush=True)
    existing = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                aid = obj.get("arxiv_id")
                if aid:
                    existing.add(aid)
            except json.JSONDecodeError:
                continue
            if (i + 1) % 500_000 == 0:
                print(f"  ... loaded {i + 1:,} lines so far", flush=True)
    print(f"Existing IDs loaded: {len(existing):,}", flush=True)
    return existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--sleep-seconds", type=float, default=3.0)
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: data/processed/arxiv_recovered.jsonl)",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    jsonl_path = config["data"]["processed_path"]
    base_url = config["data"]["oai_base_url"]
    metadata_prefix = config["data"].get("oai_metadata_prefix", "oai_dc")
    output_path = args.output or str(Path(jsonl_path).parent / "arxiv_recovered.jsonl")

    print("Recovery harvest: full undated OAI-PMH harvest (skips existing papers)", flush=True)
    print(f"Main JSONL:       {jsonl_path}", flush=True)
    print(f"Output:           {output_path}", flush=True)
    print(flush=True)

    existing_ids = load_existing_ids(jsonl_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_file = open(output_path, "w", encoding="utf-8")

    resumption_token = None
    request_count = 0
    recovered = 0
    skipped_existing = 0
    skipped_invalid = 0
    start = time.time()

    try:
        while True:
            if resumption_token:
                params = {"verb": "ListRecords", "resumptionToken": resumption_token}
            else:
                params = {
                    "verb": "ListRecords",
                    "metadataPrefix": metadata_prefix,
                }

            request_count += 1
            root = fetch_oai_xml(base_url=base_url, params=params, sleep_seconds=args.sleep_seconds)

            records = [el for el in root.iter() if local_name(el.tag) == "record"]
            for record in records:
                paper = parse_oai_record(record)
                if paper is None:
                    skipped_invalid += 1
                    continue
                if paper["arxiv_id"] in existing_ids:
                    skipped_existing += 1
                    continue
                out_file.write(json.dumps(paper, ensure_ascii=False) + "\n")
                existing_ids.add(paper["arxiv_id"])
                recovered += 1

            elapsed = time.time() - start
            print(
                f"Batch {request_count:>4}: +{recovered:>6} new  |  "
                f"skipped {skipped_existing:>7} existing, {skipped_invalid:>5} invalid  |  "
                f"{elapsed / 60:.1f}m elapsed",
                flush=True,
            )

            # Find resumption token
            resumption_token = None
            for el in root.iter():
                if local_name(el.tag) == "resumptionToken":
                    token = (el.text or "").strip()
                    resumption_token = token if token else None
                    break

            if not resumption_token:
                break

            time.sleep(args.sleep_seconds)

    finally:
        out_file.close()

    print(flush=True)
    print(f"Done. Recovered {recovered:,} new papers → {output_path}", flush=True)
    print(f"Total elapsed: {(time.time() - start) / 60:.1f}m", flush=True)
    print(flush=True)
    print("Next steps (after dense indexing finishes):", flush=True)
    print("  python scripts/merge_recovered_papers.py", flush=True)
    print("  python scripts/build_dense_index_fast.py --backend torch", flush=True)
    print("  python scripts/build_indexes.py --skip-dense", flush=True)


if __name__ == "__main__":
    main()
