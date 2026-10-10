"""
scripts/t2_prepare_data.py — Chunking logic for extracted metadata pages.

Reads structured JSON pages from data/extracted_metadata/<book>/,
then splits them into overlapping word-based chunks while combining
metadata (persons, places, dates, events) across contributing pages.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class BookChunk:
    """A single chunk of text with full metadata for the vector DB."""
    book_name: Dict[str, str]
    book_page_no: str
    content_page_no: str
    persons_mentioned: List[str]
    places_mentioned: List[str]
    dates_mentioned: List[str]
    festivals_events_mentioned: List[str]
    text_content: str
    chunk_index: int


def load_extracted_pages(book_metadata_dir: Path) -> List[Dict[str, Any]]:
    """Load all extracted metadata JSON pages of a book in order."""
    pages = []
    for json_file in sorted(book_metadata_dir.glob("page_*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("text_content"):
                pages.append(data)
    return pages


def chunk_book(
    pages: List[Dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 150,
) -> List[BookChunk]:
    """
    Split a book into overlapping chunks based on word count.

    Each chunk combines metadata (persons, places, dates, events) from
    all pages that contribute words to it.

    Args:
        pages:      List of page dicts loaded from extracted_metadata.
        chunk_size: Target number of words per chunk.
        overlap:    Number of words shared between consecutive chunks.

    Returns:
        Ordered list of BookChunk objects.
    """
    chunks = []

    # Flatten into a list of (word, page_metadata_reference)
    all_words: List[tuple] = []
    for p in pages:
        text = p.get("text_content", "")
        words = text.split()
        for w in words:
            all_words.append((w, p))

    chunk_index = 0
    i = 0
    while i < len(all_words):
        chunk_words = all_words[i : i + chunk_size]
        if not chunk_words:
            break

        text = " ".join([w[0] for w in chunk_words])

        # Find which unique pages contributed to this chunk (preserving order)
        contributing_pages: List[Dict] = []
        for w, p in chunk_words:
            if not contributing_pages or contributing_pages[-1] is not p:
                contributing_pages.append(p)

        # Union metadata across contributing pages
        book_name = contributing_pages[0].get("book_name", {})

        def unique_ordered(items):
            seen = set()
            return [x for x in items if not (x in seen or seen.add(x))]

        book_page_nos = unique_ordered(
            [p.get("book_page_no", "") for p in contributing_pages if p.get("book_page_no")]
        )
        content_page_nos = unique_ordered(
            [str(p.get("content_page_no", "")) for p in contributing_pages if p.get("content_page_no")]
        )

        persons: set = set()
        places: set = set()
        dates: set = set()
        festivals: set = set()

        for p in contributing_pages:
            persons.update(p.get("persons_mentioned", []))
            places.update(p.get("places_mentioned", []))
            dates.update(p.get("dates_mentioned", []))
            festivals.update(p.get("festivals_events_mentioned", []))

        chunks.append(
            BookChunk(
                book_name=book_name,
                book_page_no=",".join(book_page_nos),
                content_page_no=",".join(content_page_nos),
                persons_mentioned=sorted(list(persons)),
                places_mentioned=sorted(list(places)),
                dates_mentioned=sorted(list(dates)),
                festivals_events_mentioned=sorted(list(festivals)),
                text_content=text,
                chunk_index=chunk_index,
            )
        )

        chunk_index += 1
        # Move forward, keeping `overlap` words of context
        i += chunk_size - overlap

    return chunks
