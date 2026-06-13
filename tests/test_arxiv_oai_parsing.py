import xml.etree.ElementTree as ET

from scripts.update_arxiv_data import parse_authors_arxiv, parse_oai_record

NS = "http://arxiv.org/OAI/arXiv/"


def _record(metadata_xml: str, status: str = None) -> ET.Element:
    status_attr = f' status="{status}"' if status else ""
    xml = f"""
    <record>
        <header{status_attr}>
            <identifier>oai:arXiv.org:2301.07041</identifier>
            <datestamp>2023-02-14</datestamp>
        </header>
        <metadata>{metadata_xml}</metadata>
    </record>
    """
    return ET.fromstring(xml)


SAMPLE_METADATA = f"""
<arXiv xmlns="{NS}">
    <id>2301.07041</id>
    <created>2023-02-11</created>
    <updated>2023-02-14</updated>
    <authors>
        <author>
            <keyname>Viand</keyname>
            <forenames>Alexander</forenames>
        </author>
        <author>
            <keyname>Hithnawi</keyname>
        </author>
    </authors>
    <title>Verifiable Fully Homomorphic Encryption</title>
    <categories>cs.CR cs.LG</categories>
    <abstract>We analyze existing FHE integrity approaches.</abstract>
</arXiv>
"""


def test_parse_authors_handles_keyname_and_forenames():
    record = _record(SAMPLE_METADATA)
    assert parse_authors_arxiv(record) == ["Viand, Alexander", "Hithnawi"]


def test_parse_oai_record_populates_categories_and_dates():
    record = _record(SAMPLE_METADATA)
    paper = parse_oai_record(record)

    assert paper["arxiv_id"] == "2301.07041"
    assert paper["title"] == "Verifiable Fully Homomorphic Encryption"
    assert paper["abstract"] == "We analyze existing FHE integrity approaches."
    assert paper["categories"] == ["cs.CR", "cs.LG"]
    assert paper["primary_category"] == "cs.CR"
    assert paper["published_date"] == "2023-02-11"
    assert paper["updated_date"] == "2023-02-14"
    assert paper["year"] == 2023
    assert paper["url"] == "https://arxiv.org/abs/2301.07041"


def test_parse_oai_record_skips_deleted_records():
    drop_stats = {}
    record = _record(SAMPLE_METADATA, status="deleted")

    assert parse_oai_record(record, drop_stats=drop_stats) is None
    assert drop_stats == {"deleted": 1}


def test_parse_oai_record_skips_missing_abstract():
    drop_stats = {}
    metadata = f"""
    <arXiv xmlns="{NS}">
        <id>2301.07041</id>
        <created>2023-02-11</created>
        <authors><author><keyname>Viand</keyname></author></authors>
        <title>Verifiable Fully Homomorphic Encryption</title>
        <categories>cs.CR</categories>
        <abstract></abstract>
    </arXiv>
    """
    record = _record(metadata)

    assert parse_oai_record(record, drop_stats=drop_stats) is None
    assert drop_stats == {"missing_title_or_abstract": 1}


def test_parse_oai_record_falls_back_to_id_prefix_for_year():
    metadata = f"""
    <arXiv xmlns="{NS}">
        <id>2301.07041</id>
        <authors><author><keyname>Viand</keyname></author></authors>
        <title>Title</title>
        <categories>cs.CR</categories>
        <abstract>Abstract text.</abstract>
    </arXiv>
    """
    record = _record(metadata)
    paper = parse_oai_record(record)

    assert paper["year"] == 2023
    assert paper["published_date"] is None
