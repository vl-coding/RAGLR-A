import xml.etree.ElementTree as ET
from unittest.mock import patch

import pytest
import requests

from scripts.update_arxiv_data import (
    clear_checkpoint,
    fetch_oai_xml,
    harvest_oai_records,
    load_checkpoint,
)

OK_RESPONSE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    <record><header><identifier>oai:arXiv.org:0001.00001</identifier></header></record>
  </ListRecords>
</OAI-PMH>
"""


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content
        self.text = content.decode("utf-8", errors="ignore")


def test_fetch_oai_xml_retries_on_timeout_then_succeeds():
    with patch(
        "scripts.update_arxiv_data.requests.get",
        side_effect=[
            requests.exceptions.ReadTimeout("read timed out"),
            _FakeResponse(200, OK_RESPONSE_XML),
        ],
    ), patch("scripts.update_arxiv_data.time.sleep"):
        root = fetch_oai_xml("https://example.org/oai", {"verb": "ListRecords"}, max_retries=3)

    assert root.tag.endswith("OAI-PMH")


def test_fetch_oai_xml_raises_after_exhausting_retries_on_network_error():
    with patch(
        "scripts.update_arxiv_data.requests.get",
        side_effect=requests.exceptions.ConnectionError("connection reset"),
    ), patch("scripts.update_arxiv_data.time.sleep"):
        with pytest.raises(RuntimeError, match="OAI request failed after 2 attempts"):
            fetch_oai_xml("https://example.org/oai", {"verb": "ListRecords"}, max_retries=2)


def test_harvest_checkpoint_round_trip(tmp_path):
    papers_path = tmp_path / "checkpoint_papers.jsonl"
    state_path = tmp_path / "checkpoint_state.json"

    papers, token = load_checkpoint(str(papers_path), str(state_path))
    assert papers == []
    assert token is None

    with patch(
        "scripts.update_arxiv_data.fetch_oai_xml",
        return_value=ET.fromstring(
            """
            <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
              <ListRecords>
                <record>
                  <header><identifier>oai:arXiv.org:2301.00001</identifier></header>
                  <metadata>
                    <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
                      <id>2301.00001</id>
                      <created>2023-01-01</created>
                      <authors><author><keyname>Doe</keyname></author></authors>
                      <title>A Paper</title>
                      <categories>cs.AI</categories>
                      <abstract>An abstract.</abstract>
                    </arXiv>
                  </metadata>
                </record>
              </ListRecords>
            </OAI-PMH>
            """
        ),
    ), patch("scripts.update_arxiv_data.time.sleep"):
        harvested = harvest_oai_records(
            base_url="https://example.org/oai",
            metadata_prefix="arXiv",
            checkpoint_papers_path=str(papers_path),
            checkpoint_state_path=str(state_path),
        )

    assert len(harvested) == 1
    assert papers_path.exists()

    checkpoint_papers, resumption_token = load_checkpoint(str(papers_path), str(state_path))
    assert len(checkpoint_papers) == 1
    assert checkpoint_papers[0]["arxiv_id"] == "2301.00001"
    assert resumption_token is None

    clear_checkpoint(str(papers_path), str(state_path))
    assert not papers_path.exists()
    assert not state_path.exists()
