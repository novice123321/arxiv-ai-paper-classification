"""Collect arXiv computer science paper metadata and save it as CSV."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import requests
from lxml import etree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_OUTPUT = DATA_DIR / "raw_arxiv.csv"

API_URL = "https://export.arxiv.org/api/query"
CATEGORIES = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CL": "Computation and Language",
    "cs.SE": "Software Engineering",
}

RESULTS_PER_CATEGORY = 5000
PAGE_SIZE = 100
REQUEST_DELAY_SECONDS = 3.2
TIMEOUT_SECONDS = 30

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def normalize_text(value: str | None) -> str:
    """Collapse repeated whitespace in text fields returned by arXiv."""
    if not value:
        return ""
    return " ".join(value.split())


def build_query_url(category: str, start: int, max_results: int) -> str:
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return f"{API_URL}?{urlencode(params)}"


def fetch_xml(category: str, start: int, max_results: int) -> bytes:
    """Fetch one arXiv API page with status-code and exception handling."""
    url = build_query_url(category, start, max_results)
    headers = {
        "User-Agent": (
            "DataEngineeringCourseProject/1.0 "
            "(student project; contact: 2023100255@example.edu)"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch {category} start={start}: {exc}") from exc
    return response.content


def extract_entry_url(entry: etree._Element) -> tuple[str, str]:
    """Return the abstract page URL and PDF URL from an arXiv entry."""
    entry_url = ""
    pdf_url = ""
    for link in entry.xpath("atom:link", namespaces=NS):
        href = link.get("href", "")
        rel = link.get("rel", "")
        title = link.get("title", "")
        if rel == "alternate":
            entry_url = href
        if title == "pdf" or href.endswith(".pdf"):
            pdf_url = href
    return entry_url, pdf_url


def parse_entries(xml_bytes: bytes, source_category: str) -> list[dict[str, str]]:
    """Parse Atom XML using XPath and return normalized paper records."""
    root = etree.fromstring(xml_bytes)
    rows: list[dict[str, str]] = []

    for entry in root.xpath("//atom:entry", namespaces=NS):
        entry_id = normalize_text(entry.xpath("string(atom:id)", namespaces=NS))
        title = normalize_text(entry.xpath("string(atom:title)", namespaces=NS))
        summary = normalize_text(entry.xpath("string(atom:summary)", namespaces=NS))
        published = normalize_text(entry.xpath("string(atom:published)", namespaces=NS))
        updated = normalize_text(entry.xpath("string(atom:updated)", namespaces=NS))
        authors = [
            normalize_text(name)
            for name in entry.xpath("atom:author/atom:name/text()", namespaces=NS)
        ]
        primary_category = normalize_text(
            entry.xpath("string(arxiv:primary_category/@term)", namespaces=NS)
        )
        categories = [
            normalize_text(term)
            for term in entry.xpath("atom:category/@term", namespaces=NS)
        ]
        entry_url, pdf_url = extract_entry_url(entry)
        arxiv_id = entry_id.rsplit("/", 1)[-1]

        rows.append(
            {
                "source_category": source_category,
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": summary,
                "authors": "; ".join(authors),
                "published": published,
                "updated": updated,
                "primary_category": primary_category,
                "categories": "; ".join(categories),
                "entry_url": entry_url,
                "pdf_url": pdf_url,
            }
        )
    return rows


def collect_category(category: str) -> list[dict[str, str]]:
    """Collect one category in several API pages."""
    collected: list[dict[str, str]] = []
    for start in range(0, RESULTS_PER_CATEGORY, PAGE_SIZE):
        page_size = min(PAGE_SIZE, RESULTS_PER_CATEGORY - start)
        print(f"Fetching {category}: start={start}, max_results={page_size}")
        xml_bytes = fetch_xml(category, start, page_size)
        rows = parse_entries(xml_bytes, source_category=category)
        collected.extend(rows)
        if start + page_size < RESULTS_PER_CATEGORY:
            time.sleep(REQUEST_DELAY_SECONDS)
    return collected


def write_csv(rows: Iterable[dict[str, str]], output_path: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_category",
        "arxiv_id",
        "title",
        "summary",
        "authors",
        "published",
        "updated",
        "primary_category",
        "categories",
        "entry_url",
        "pdf_url",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_rows: list[dict[str, str]] = []
    for category in CATEGORIES:
        all_rows.extend(collect_category(category))
        time.sleep(REQUEST_DELAY_SECONDS)

    unique_rows = {row["arxiv_id"]: row for row in all_rows if row["arxiv_id"]}
    write_csv(unique_rows.values(), RAW_OUTPUT)
    print(f"Saved {len(unique_rows)} unique records to {RAW_OUTPUT}")


if __name__ == "__main__":
    main()
