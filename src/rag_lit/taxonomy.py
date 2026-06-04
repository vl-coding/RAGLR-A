from typing import Dict, List, Optional

import yaml


def load_taxonomy(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def all_category_codes(taxonomy: dict) -> List[str]:
    """Returns every concrete arXiv category code from the taxonomy."""
    codes = set()
    for field_data in taxonomy.get("fields", {}).values():
        cats = field_data.get("categories", [])
        if cats == "*" or not isinstance(cats, list):
            continue
        for cat in cats:
            codes.add(cat)
    return sorted(codes)


def field_label_map(taxonomy: dict) -> Dict[str, str]:
    """Returns {field_key: label} for every field in the taxonomy."""
    return {
        key: data["label"]
        for key, data in taxonomy.get("fields", {}).items()
    }


def categories_for_field(taxonomy: dict, field_key: str) -> Optional[List[str]]:
    """Returns category codes for a given field key, or None if it matches all."""
    field_data = taxonomy.get("fields", {}).get(field_key)
    if field_data is None:
        raise ValueError(f"Unknown field: {field_key}")
    cats = field_data.get("categories", [])
    if cats == "*":
        return None
    return list(cats)
