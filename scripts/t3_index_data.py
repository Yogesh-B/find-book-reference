#!/usr/bin/env python3
"""
scripts/t3_index_data.py — Chunk extracted metadata and index into Qdrant vector DB.

Reads structured JSON pages from data/extracted_metadata/<book>/,
uses t2_prepare_data.py to chunk them, generates multilingual embeddings
(intfloat/multilingual-e5-large), and upserts into a local Qdrant collection.

Usage (from project root, with venv active):

  # Index all books that have extracted metadata:
  python scripts/t3_index_data.py

  # Index a single book only:
  python scripts/t3_index_data.py --book 72181_Shri_Harikrushna_Charitramrut_Sagar_Bhag_1

  # Re-index (drop and recreate the collection first):
  python scripts/t3_index_data.py --reindex

  # Custom chunk size / overlap (words):
  python scripts/t3_index_data.py --chunk-size 600 --overlap 150

  # Custom Qdrant path or server:
  python scripts/t3_index_data.py --qdrant-path ./data/vector_db
  python scripts/t3_index_data.py --qdrant-host localhost --qdrant-port 6333

Output:
  data/vector_db/   — Qdrant local storage (embedded mode)
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("t3_index_data")

# ---------------------------------------------------------------------------
# Imports (graceful error messages)
# ---------------------------------------------------------------------------

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        HnswConfigDiff,
        MatchValue,
        OptimizersConfigDiff,
        PointStruct,
        VectorParams,
    )
except ImportError:
    sys.exit("❌  qdrant-client not installed. Run: pip install qdrant-client")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("❌  sentence-transformers not installed. Run: pip install sentence-transformers")

# Allow running from both project root and scripts/ directory
sys.path.insert(0, str(Path(__file__).parent))
try:
    from t2_prepare_data import BookChunk, chunk_book, load_extracted_pages
except ImportError:
    sys.exit("❌  Could not import t2_prepare_data. Run from project root.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "gujarati_scriptures"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 64          # Points per upsert call
DEFAULT_CHUNK_SIZE = 500  # Words per chunk
DEFAULT_OVERLAP = 150     # Overlap words between chunks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def iter_batches(items: list, size: int) -> Iterator[list]:
    """Yield successive fixed-size batches from a list."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def chunk_id(book_english: str, chunk_index: int) -> str:
    """Deterministic UUID for a chunk so re-indexing is idempotent (upsert)."""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"{book_english}::chunk_{chunk_index:06d}",
        )
    )


def chunk_to_point(chunk: BookChunk, embedding: list[float]) -> PointStruct:
    """Convert a BookChunk + embedding into a Qdrant PointStruct."""
    book_english = chunk.book_name.get("english", "unknown")
    return PointStruct(
        id=chunk_id(book_english, chunk.chunk_index),
        vector=embedding,
        payload={
            "book_name": chunk.book_name,
            "book_page_no": chunk.book_page_no,
            "content_page_no": chunk.content_page_no,
            "persons_mentioned": chunk.persons_mentioned,
            "places_mentioned": chunk.places_mentioned,
            "dates_mentioned": chunk.dates_mentioned,
            "festivals_events_mentioned": chunk.festivals_events_mentioned,
            "text_content": chunk.text_content,
            "chunk_index": chunk.chunk_index,
        },
    )


# ---------------------------------------------------------------------------
# Core indexing logic
# ---------------------------------------------------------------------------


def build_client(args: argparse.Namespace) -> QdrantClient:
    if args.qdrant_host:
        logger.info("Connecting to Qdrant at %s:%s", args.qdrant_host, args.qdrant_port)
        return QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    qdrant_path = Path(args.qdrant_path)
    qdrant_path.mkdir(parents=True, exist_ok=True)
    logger.info("Using local Qdrant at %s", qdrant_path)
    return QdrantClient(path=str(qdrant_path))


def ensure_collection(client: QdrantClient, vector_dim: int, reindex: bool) -> None:
    exists = client.collection_exists(COLLECTION_NAME)
    if exists and reindex:
        logger.warning("--reindex: dropping existing collection '%s'", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if not exists:
        logger.info("Creating collection '%s' (dim=%d, COSINE)", COLLECTION_NAME, vector_dim)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20_000,
            ),
        )
    else:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(
            "Collection '%s' already exists with %d vectors — will upsert (no duplicates).",
            COLLECTION_NAME,
            info.vectors_count or 0,
        )


def index_book(
    book_dir: Path,
    embed_model: SentenceTransformer,
    client: QdrantClient,
    chunk_size: int,
    overlap: int,
) -> int:
    """Load, chunk, embed and index one book. Returns number of chunks indexed."""
    book_name = book_dir.name
    pages = load_extracted_pages(book_dir)
    if not pages:
        logger.warning("  [%s] No extracted pages found — skipping.", book_name)
        return 0

    chunks = chunk_book(pages, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        logger.warning("  [%s] No chunks generated — skipping.", book_name)
        return 0

    logger.info("  [%s] %d pages → %d chunks", book_name, len(pages), len(chunks))

    total_upserted = 0
    for batch_chunks in iter_batches(chunks, BATCH_SIZE):
        texts = [f"passage: {c.text_content}" for c in batch_chunks]
        embeddings = embed_model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        points = [
            chunk_to_point(chunk, emb.tolist())
            for chunk, emb in zip(batch_chunks, embeddings)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_upserted += len(points)

    return total_upserted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chunk extracted metadata and index into Qdrant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--book",
        metavar="BOOK_FOLDER",
        default=None,
        help="Index only this book folder (e.g. 72181_Shri_Harikrushna_Charitramrut_Sagar_Bhag_1). "
             "Omit to process all books in data/extracted_metadata/.",
    )
    p.add_argument(
        "--reindex",
        action="store_true",
        help="Drop and recreate the Qdrant collection before indexing.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Target number of words per chunk.",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Number of words to overlap between consecutive chunks.",
    )
    p.add_argument(
        "--qdrant-path",
        default="./data/vector_db",
        help="Path for Qdrant local embedded storage.",
    )
    p.add_argument(
        "--qdrant-host",
        default=None,
        help="Qdrant server host (use instead of --qdrant-path for a running server).",
    )
    p.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant server port (used with --qdrant-host).",
    )
    p.add_argument(
        "--metadata-dir",
        default="./data/extracted_metadata",
        help="Root directory containing per-book extracted metadata.",
    )
    p.add_argument(
        "--embed-model",
        default=EMBED_MODEL_NAME,
        help="SentenceTransformer model name for embeddings.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    metadata_root = Path(args.metadata_dir)
    if not metadata_root.exists():
        sys.exit(f"❌  Metadata directory not found: {metadata_root}")

    # Determine which books to process
    if args.book:
        book_dirs = [metadata_root / args.book]
        if not book_dirs[0].is_dir():
            sys.exit(f"❌  Book not found in metadata dir: {book_dirs[0]}")
    else:
        book_dirs = sorted(
            d for d in metadata_root.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )
        if not book_dirs:
            sys.exit(f"❌  No book directories found under {metadata_root}")

    logger.info("Books to index: %d", len(book_dirs))
    for d in book_dirs:
        page_count = len(list(d.glob("page_*.json")))
        logger.info("  %s (%d pages)", d.name, page_count)

    # Load embedding model
    logger.info("Loading embedding model: %s", args.embed_model)
    embed_model = SentenceTransformer(args.embed_model)
    vector_dim = embed_model.get_sentence_embedding_dimension()
    logger.info("Embedding dimension: %d", vector_dim)

    # Connect to Qdrant and prepare collection
    client = build_client(args)
    ensure_collection(client, vector_dim, args.reindex)

    # Index each book
    grand_total = 0
    for book_dir in book_dirs:
        logger.info("Processing: %s", book_dir.name)
        count = index_book(
            book_dir=book_dir,
            embed_model=embed_model,
            client=client,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        logger.info("  → %d chunks indexed", count)
        grand_total += count

    # Final summary
    info = client.get_collection(COLLECTION_NAME)
    logger.info("")
    logger.info("✅  Indexing complete!")
    logger.info("   Total chunks indexed this run : %d", grand_total)
    vec_count = getattr(info, "points_count", None) or getattr(info, "vectors_count", "?")
    logger.info("   Total vectors in collection   : %s", vec_count)
    logger.info("   Collection                    : %s", COLLECTION_NAME)
    logger.info("   Qdrant path                   : %s", args.qdrant_path)


if __name__ == "__main__":
    main()
