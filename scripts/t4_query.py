#!/usr/bin/env python3
"""
scripts/t4_query.py — Semantic search & RAG query pipeline for Gujarati scriptures.

Searches the local Qdrant vector database (created by t3_index_data.py) using
multilingual dense embeddings (intfloat/multilingual-e5-large), retrieves the
most relevant scripture chunks with exact source citations (book name, chapter,
book page, content page), and optionally synthesizes an answer via an LLM
(local OpenAI-compatible server or Antigravity CLI).

Usage (from project root, with .venv active):

  # 1. Basic vector search & retrieval (search-only):
  python scripts/t4_query.py -q "ભાડેર ગામનો રેફરન્સ ક્યાં આવે છે?" --no-llm

  # 2. Filter search by specific book, person, or place:
  python scripts/t4_query.py -q "પાતળભાઈની વાત" --place "ભાડેર" --no-llm
  python scripts/t4_query.py -q "ચમત્કાર" --book "Shri Harikrushna Charitramrut Sagar Bhag 2"

  # 3. Full RAG: retrieve chunks + answer via local LLM (e.g. llama-server):
  python scripts/t4_query.py -q "ભાડેરના પાતળભાઈનો પ્રસંગ શું છે?" \
      --openai-url http://localhost:8080/v1

  # 4. Full RAG via Antigravity CLI (agy):
  python scripts/t4_query.py -q "ભાડેરના પાતળભાઈનો પ્રસંગ શું છે?" \
      --backend agy

  # 5. Interactive Q&A loop (loads embedding model once, asks repeatedly):
  python scripts/t4_query.py --interactive

  # 6. JSON output for API / Web UI integration:
  python scripts/t4_query.py -q "ભાડેર ગામ" --no-llm --json

Arguments & Options:
  Flexible paths and parameters are supported without hardcoding. Run with --help
  for the complete options list.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("t4_query")

# ---------------------------------------------------------------------------
# Imports (with graceful failure instructions)
# ---------------------------------------------------------------------------

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        MatchText,
        MatchValue,
    )
except ImportError:
    sys.exit("❌ qdrant-client not installed. Run: pip install qdrant-client")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("❌ sentence-transformers not installed. Run: pip install sentence-transformers")


# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION = "gujarati_scriptures"
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_QDRANT_PATH = "./data/vector_db"
DEFAULT_OPENAI_URL = "http://localhost:8080/v1"
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.35

RAG_SYSTEM_PROMPT = """\
You are an authoritative and precise Gujarati scripture expert assistant.
You answer questions based ONLY on the provided context retrieved from authentic scriptures.

Strict Rules:
1. Always cite the exact source in your answer:
   - Book Name in Gujarati and English
   - Book Page Number and Content Page Number
2. Answer in pure Gujarati when the question is in Gujarati.
3. Quote relevant lines / shlokas / kirtan verses from the source text when appropriate.
4. If the context does not contain enough information to answer the question, clearly state:
   "આ વિષયે આપેલા સંદર્ભોમાં માહિતી મળતી નથી." (Information not found in the provided references).
5. Never invent or hallucinate dates, persons, places, or events not present in the context.
6. Be concise, respectful, and factually exact.
"""


# ---------------------------------------------------------------------------
# Terminal Color Formatting
# ---------------------------------------------------------------------------

class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def colorize(text: str, color: str, enabled: bool = True) -> str:
    if not enabled:
        return text
    return f"{color}{text}{Colors.RESET}"


# ---------------------------------------------------------------------------
# Query Engine Core
# ---------------------------------------------------------------------------

class ScriptureQueryEngine:
    """Manages vector search retrieval and LLM answer generation."""

    def __init__(
        self,
        qdrant_path: str = DEFAULT_QDRANT_PATH,
        qdrant_host: Optional[str] = None,
        qdrant_port: int = 6333,
        collection_name: str = DEFAULT_COLLECTION,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        device: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.qdrant_client = self._init_qdrant(qdrant_path, qdrant_host, qdrant_port)
        self.embed_model_name = embed_model_name
        self.device = device
        self._embed_model: Optional[SentenceTransformer] = None

    def _init_qdrant(
        self,
        qdrant_path: str,
        qdrant_host: Optional[str],
        qdrant_port: int,
    ) -> QdrantClient:
        if qdrant_host:
            logger.info("Connecting to remote Qdrant at %s:%s", qdrant_host, qdrant_port)
            return QdrantClient(host=qdrant_host, port=qdrant_port)

        path_obj = Path(qdrant_path).resolve()
        if not path_obj.exists():
            logger.warning("Vector DB directory does not exist yet at %s", path_obj)
        logger.info("Connecting to local Qdrant at %s", path_obj)
        return QdrantClient(path=str(path_obj))

    @property
    def embed_model(self) -> SentenceTransformer:
        """Lazy load embedding model so CLI can validate parameters quickly."""
        if self._embed_model is None:
            logger.info("Loading embedding model '%s'...", self.embed_model_name)
            self._embed_model = SentenceTransformer(
                self.embed_model_name,
                device=self.device,
            )
            logger.info("Embedding model loaded successfully.")
        return self._embed_model

    def build_filter(
        self,
        book: Optional[str] = None,
        person: Optional[str] = None,
        place: Optional[str] = None,
    ) -> Optional[Filter]:
        """Construct a Qdrant Filter based on metadata criteria."""
        conditions = []

        if book:
            # Match either English or Gujarati title
            book_filter = Filter(
                should=[
                    FieldCondition(key="book_name.english", match=MatchValue(value=book)),
                    FieldCondition(key="book_name.gujarati", match=MatchValue(value=book)),
                    FieldCondition(key="book_name.english", match=MatchText(text=book)),
                    FieldCondition(key="book_name.gujarati", match=MatchText(text=book)),
                ]
            )
            conditions.append(book_filter)

        if person:
            conditions.append(
                FieldCondition(key="persons_mentioned", match=MatchValue(value=person))
            )

        if place:
            conditions.append(
                FieldCondition(key="places_mentioned", match=MatchValue(value=place))
            )

        if not conditions:
            return None

        return Filter(must=conditions)

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: Optional[float] = DEFAULT_SCORE_THRESHOLD,
        book_filter: Optional[str] = None,
        person_filter: Optional[str] = None,
        place_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed query and retrieve top matching chunks from Qdrant.
        E5 models require prefix: 'query: <text>'.
        """
        if not query.strip():
            return []

        # Prefix for e5 models
        e5_query = f"query: {query.strip()}"
        query_vector = self.embed_model.encode(
            e5_query,
            normalize_embeddings=True,
        ).tolist()

        query_filter = self.build_filter(
            book=book_filter,
            person=person_filter,
            place=place_filter,
        )

        # Support both newer query_points() and older search() APIs
        if hasattr(self.qdrant_client, "query_points"):
            res = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
            )
            hits = res.points
        else:
            hits = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
            )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "id": str(hit.id),
                "score": float(hit.score) if hit.score is not None else 0.0,
                "book_name": payload.get("book_name", {}),
                "book_page_no": payload.get("book_page_no", ""),
                "content_page_no": payload.get("content_page_no", ""),
                "persons_mentioned": payload.get("persons_mentioned", []),
                "places_mentioned": payload.get("places_mentioned", []),
                "dates_mentioned": payload.get("dates_mentioned", []),
                "festivals_events_mentioned": payload.get("festivals_events_mentioned", []),
                "text_content": payload.get("text_content", ""),
                "chunk_index": payload.get("chunk_index", 0),
            })

        return results

    def build_context(self, chunks: List[Dict[str, Any]], max_chunks: int = 5) -> str:
        """Format retrieved chunks with citation headers for the LLM."""
        context_parts = []
        for i, c in enumerate(chunks[:max_chunks], 1):
            b = c.get("book_name", {})
            bguj = b.get("gujarati", "અજ્ઞાત પુસ્તક")
            beng = b.get("english", "Unknown Book")
            bp = c.get("book_page_no", "-")
            cp = c.get("content_page_no", "-")
            persons = ", ".join(c.get("persons_mentioned", [])[:8]) or "કોઈ નહીં"
            places = ", ".join(c.get("places_mentioned", [])[:8]) or "કોઈ નહીં"

            header = (
                f"--- [સંદર્ભ {i}] પુસ્તક: {bguj} ({beng}) | "
                f"બુક પાના નં: {bp} | કન્ટેન્ટ પાના નં: {cp} ---\n"
                f"ઉલ્લેખિત વ્યક્તિઓ: {persons}\n"
                f"ઉલ્લેખિત સ્થાનો: {places}"
            )
            context_parts.append(f"{header}\n\n{c.get('text_content', '').strip()}\n")

        return "\n".join(context_parts)

    def ask_llm_openai(
        self,
        question: str,
        context: str,
        openai_url: str = DEFAULT_OPENAI_URL,
        model_name: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        """Call an OpenAI-compatible completion endpoint (e.g. llama-server)."""
        url = openai_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"

        user_content = (
            f"સંદર્ભ લખાણ (Context from Scripture Database):\n\n"
            f"{context}\n\n"
            f"પ્રશ્ન (Question): {question}\n\n"
            f"વિગતવાર જવાબ (ચોક્કસ પુસ્તક, પ્રકરણ અને પાના નંબર ટાંકીને જવાબ આપો):"
        )

        body = {
            "model": model_name or "default",
            "messages": [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        req_data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                return resp_json["choices"][0]["message"]["content"].strip()
        except urllib.error.URLError as e:
            return f"❌ OpenAI API Connection Error ({url}): {e}"
        except Exception as e:
            return f"❌ Failed to parse LLM response: {e}"

    def ask_llm_agy(
        self,
        question: str,
        context: str,
        model_name: Optional[str] = None,
    ) -> str:
        """Call Antigravity CLI (agy) via subprocess."""
        if not shutil.which("agy"):
            return "❌ 'agy' command not found in PATH."

        prompt = (
            f"{RAG_SYSTEM_PROMPT}\n\n"
            f"સંદર્ભ લખાણ (Context):\n{context}\n\n"
            f"પ્રશ્ન (Question): {question}\n\n"
            f"વિગતવાર જવાબ (પુસ્તક અને પાના નંબર સાથે):"
        )

        cmd = ["agy", "-p", prompt]
        if model_name:
            cmd.extend(["--model", model_name])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=180,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"❌ agy returned error: {e.stderr.strip()}"
        except Exception as e:
            return f"❌ Failed calling agy: {e}"


# ---------------------------------------------------------------------------
# Terminal Output Presentation
# ---------------------------------------------------------------------------

def display_results(
    query: str,
    chunks: List[Dict[str, Any]],
    llm_answer: Optional[str] = None,
    use_color: bool = True,
) -> None:
    """Print readable and structured search results with source citations."""
    print()
    print(colorize("=" * 72, Colors.BLUE, use_color))
    print(colorize(f"🔍 પ્રશ્ન / Query: {query}", Colors.BOLD, use_color))
    print(colorize("=" * 72, Colors.BLUE, use_color))

    if not chunks:
        print(colorize("\n⚠️  કોઈ સંબંધિત રેફરન્સ મળ્યા નથી. (No matching passages found)", Colors.YELLOW, use_color))
        return

    print(colorize(f"\n📚 મળેલા રેફરન્સ (Found {len(chunks)} relevant passages):\n", Colors.GREEN, use_color))

    for i, c in enumerate(chunks, 1):
        b = c.get("book_name", {})
        bguj = b.get("gujarati", "અજ્ઞાત")
        beng = b.get("english", "Unknown")
        bp = c.get("book_page_no", "-")
        cp = c.get("content_page_no", "-")
        score = c.get("score", 0.0)
        persons = c.get("persons_mentioned", [])
        places = c.get("places_mentioned", [])
        txt = c.get("text_content", "").strip()

        print(colorize(f"[{i}] સ્કોર (Similarity): {score:.4f}", Colors.CYAN, use_color))
        print(f"    📖 પુસ્તક: {colorize(bguj, Colors.BOLD, use_color)} ({beng})")
        print(f"    📄 બુક પેજ નં: {colorize(bp, Colors.YELLOW, use_color)} | કન્ટેન્ટ પેજ નં: {colorize(cp, Colors.YELLOW, use_color)}")
        if persons:
            print(f"    👤 વ્યક્તિઓ: {', '.join(persons[:10])}")
        if places:
            print(f"    📍 સ્થાનો: {', '.join(places[:10])}")

        snippet = txt[:280].replace("\n", " ")
        print(colorize(f"    💬 લખાણ: \"{snippet}...\"", Colors.DIM, use_color))
        print(colorize("-" * 72, Colors.DIM, use_color))

    if llm_answer:
        print(colorize("\n🤖 સંક્ષિપ્ત ઉત્તર (LLM Answer with Citations):", Colors.BOLD + Colors.GREEN, use_color))
        print(colorize("-" * 72, Colors.GREEN, use_color))
        print(llm_answer)
        print(colorize("-" * 72, Colors.GREEN, use_color))

    print()


# ---------------------------------------------------------------------------
# Interactive Shell Mode
# ---------------------------------------------------------------------------

def run_interactive_loop(
    engine: ScriptureQueryEngine,
    args: argparse.Namespace,
    use_color: bool = True,
) -> None:
    """Keep embedding model loaded and allow continuous user queries."""
    print(colorize("\n📖 ગુજરાતી શાસ્ત્ર રેફરન્સ ફાઇન્ડર — ઇન્ટરેક્ટિવ મોડ (Interactive Mode)", Colors.BOLD + Colors.CYAN, use_color))
    print("પ્રશ્ન ટાઈપ કરો અને Enter દબાવો (બહાર નીકળવા માટે 'exit' અથવા 'quit' લખો):\n")

    while True:
        try:
            query = input(colorize("સવાલ (Query) > ", Colors.YELLOW, use_color)).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nઆભાર!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("આભાર!")
            break

        chunks = engine.search(
            query=query,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            book_filter=args.book,
            person_filter=args.person,
            place_filter=args.place,
        )

        llm_answer = None
        if not args.no_llm and chunks:
            context = engine.build_context(chunks, max_chunks=args.max_context_chunks)
            if args.backend == "agy":
                llm_answer = engine.ask_llm_agy(
                    question=query,
                    context=context,
                    model_name=args.model,
                )
            else:
                llm_answer = engine.ask_llm_openai(
                    question=query,
                    context=context,
                    openai_url=args.openai_url,
                    model_name=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )

        display_results(query, chunks, llm_answer, use_color=use_color)


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic search and RAG citation pipeline for Gujarati scriptures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Query Input
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Search query or question (in Gujarati or English). Omit for interactive mode.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch interactive Q&A shell loop.",
    )

    # Retrieval Configuration
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Maximum number of relevant chunks to retrieve.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="Minimum cosine similarity score (0.0 to 1.0).",
    )
    parser.add_argument(
        "--max-context-chunks",
        type=int,
        default=5,
        help="Maximum top chunks forwarded to LLM prompt.",
    )

    # Metadata Filters
    parser.add_argument(
        "--book",
        type=str,
        default=None,
        help="Filter search to a specific book title (English or Gujarati).",
    )
    parser.add_argument(
        "--person",
        type=str,
        default=None,
        help="Filter results to chunks mentioning this person.",
    )
    parser.add_argument(
        "--place",
        type=str,
        default=None,
        help="Filter results to chunks mentioning this place / village.",
    )

    # Storage & Embedding Configuration (Flexible, not hardcoded)
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default=DEFAULT_QDRANT_PATH,
        help="Local directory path of embedded Qdrant database.",
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default=None,
        help="Qdrant server host (use for remote/standalone container).",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant server port.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="Qdrant collection name to search.",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=DEFAULT_EMBED_MODEL,
        help="SentenceTransformer model name for generating query embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device for embedding model (e.g. 'cpu', 'cuda').",
    )

    # LLM Synthesis Configuration
    parser.add_argument(
        "--no-llm", "--search-only",
        action="store_true",
        dest="no_llm",
        help="Run vector retrieval only without calling an LLM.",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "agy"],
        default="openai",
        help="LLM backend: 'openai' for local OpenAI-compatible server, or 'agy' for Antigravity CLI.",
    )
    parser.add_argument(
        "--openai-url",
        type=str,
        default=DEFAULT_OPENAI_URL,
        help="Base URL for OpenAI-compatible LLM endpoint (e.g. llama-server).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier passed to the LLM backend.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for LLM generation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens generated by LLM.",
    )

    # Output Formatting
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON structure instead of terminal formatted text.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in terminal output.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    use_color = not args.no_color and sys.stdout.isatty() and not args.json

    engine = ScriptureQueryEngine(
        qdrant_path=args.qdrant_path,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection,
        embed_model_name=args.embed_model,
        device=args.device,
    )

    # Enter interactive shell if requested or if no query is given
    if args.interactive or not args.query:
        run_interactive_loop(engine, args, use_color=use_color)
        return

    # Single Query Execution
    chunks = engine.search(
        query=args.query,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        book_filter=args.book,
        person_filter=args.person,
        place_filter=args.place,
    )

    llm_answer = None
    if not args.no_llm and chunks:
        context = engine.build_context(chunks, max_chunks=args.max_context_chunks)
        if args.backend == "agy":
            llm_answer = engine.ask_llm_agy(
                question=args.query,
                context=context,
                model_name=args.model,
            )
        else:
            llm_answer = engine.ask_llm_openai(
                question=args.query,
                context=context,
                openai_url=args.openai_url,
                model_name=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

    if args.json:
        output_payload = {
            "query": args.query,
            "results_count": len(chunks),
            "chunks": chunks,
            "llm_answer": llm_answer,
        }
        print(json.dumps(output_payload, ensure_ascii=False, indent=2))
    else:
        display_results(args.query, chunks, llm_answer, use_color=use_color)


if __name__ == "__main__":
    main()
