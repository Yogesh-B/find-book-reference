#!/usr/bin/env python3
"""
scripts/extract_metadata.py — LLM-powered metadata extraction for Gujarati scripture OCR pages.

Reads OCR markdown and JSON pages, builds prompts with contextual hints,
calls Antigravity CLI (agy) or a local OpenAI-compatible endpoint, validates
the output against page_metadata_schema.json, and saves one structured JSON
file per page under data/extracted_metadata/<book>/.

Usage (from project root):

  # Extract metadata for a single book (default: agy backend):
  python scripts/extract_metadata.py --book 72181_Shri_Harikrushna_Charitramrut_Sagar_Bhag_1

  # Extract specific pages only:
  python scripts/extract_metadata.py --book <book_folder> --pages 1-50
  python scripts/extract_metadata.py --book <book_folder> --pages 1,5,10-20

  # Resume a previous run (skip already-extracted pages):
  python scripts/extract_metadata.py --book <book_folder> --resume

  # Process all books in data/ocr_output/:
  python scripts/extract_metadata.py --all

  # Run 4 pages in parallel with a faster model:
  python scripts/extract_metadata.py --book <book_folder> --concurrency 4 --model gemini-3.8-flash-low

  # Use a local OpenAI-compatible server (e.g. llama-server):
  python scripts/extract_metadata.py --book <book_folder> \\
      --use-openai --openai-url http://localhost:8080

  # Dry run — see what would be processed without calling the LLM:
  python scripts/extract_metadata.py --book <book_folder> --dry-run

Output layout:
  data/extracted_metadata/
    <book_folder>/
      page_0001.json   # one file per page
      page_0002.json
      ...
    _progress.json     # tracks processed/failed pages per book
    _summary.json      # aggregate summary across all books
    _errors/<book>/    # error logs for pages that failed all retries
"""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request

# Import prompt builder from local module
try:
    from prompts import build_prompt, OPENAI_SYSTEM_PROMPT
except ImportError:
    from scripts.prompts import build_prompt, OPENAI_SYSTEM_PROMPT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_metadata")


class ProgressTracker:
    """Thread-safe progress and summary tracker stored in output directory."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.progress_file = output_dir / "_progress.json"
        self.summary_file = output_dir / "_summary.json"
        self.lock = threading.Lock()
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}. Starting fresh.")
        return {}

    def is_processed(self, book_name: str, page_num: int) -> bool:
        with self.lock:
            book_info = self.data.get(book_name, {})
            return page_num in book_info.get("processed_pages", [])

    def mark_processed(self, book_name: str, page_num: int, total_pages: int) -> None:
        with self.lock:
            book_info = self.data.setdefault(
                book_name,
                {"total_pages": total_pages, "processed_pages": [], "failed_pages": []},
            )
            book_info["total_pages"] = total_pages
            if page_num not in book_info["processed_pages"]:
                book_info["processed_pages"].append(page_num)
                book_info["processed_pages"].sort()
            if page_num in book_info.get("failed_pages", []):
                book_info["failed_pages"].remove(page_num)
            book_info["last_updated"] = datetime.now(timezone.utc).isoformat()
            self._save()

    def mark_failed(self, book_name: str, page_num: int, total_pages: int) -> None:
        with self.lock:
            book_info = self.data.setdefault(
                book_name,
                {"total_pages": total_pages, "processed_pages": [], "failed_pages": []},
            )
            book_info["total_pages"] = total_pages
            if page_num not in book_info.get("failed_pages", []):
                book_info["failed_pages"].append(page_num)
                book_info["failed_pages"].sort()
            book_info["last_updated"] = datetime.now(timezone.utc).isoformat()
            self._save()

    def _save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temp_file = self.progress_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            temp_file.replace(self.progress_file)
        except Exception as e:
            logger.error(f"Error saving progress file: {e}")

    def save_summary(self) -> None:
        with self.lock:
            total_books = len(self.data)
            total_pages_all = sum(b.get("total_pages", 0) for b in self.data.values())
            processed_all = sum(len(b.get("processed_pages", [])) for b in self.data.values())
            failed_all = sum(len(b.get("failed_pages", [])) for b in self.data.values())

            summary = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_books_tracked": total_books,
                "total_pages_across_tracked_books": total_pages_all,
                "total_pages_processed": processed_all,
                "total_pages_failed": failed_all,
                "books": self.data,
            }
            self.output_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.summary_file, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Error saving summary file: {e}")


def clean_text_content(text: str) -> str:
    """
    Remove page number markers (e.g. $$84$$, $૮૫$, $$ 12 $$) and normalize whitespace.
    """
    if not text:
        return ""
    # Strip markers like $$84$$, $૮૫$, $$ 12 $$
    cleaned = re.sub(r"\$\$?\s*[\d૦-૯]+\s*\$\$?", "", text)
    # Strip standalone page marker lines if any remaining
    lines = [line.strip() for line in cleaned.splitlines()]
    # Normalize multiple blank lines into single blank line
    output_lines: list[str] = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                output_lines.append("")
                prev_blank = True
        else:
            output_lines.append(line)
            prev_blank = False
    return "\n".join(output_lines).strip()


def load_page_text(book_dir: Path, page_num: int) -> str | None:
    """Read markdown page text for a given page number."""
    md_file = book_dir / "raw" / "markdown" / f"page_{page_num:04d}.md"
    if not md_file.exists():
        return None
    try:
        return md_file.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read {md_file}: {e}")
        return None


def load_ocr_json(book_dir: Path, page_num: int) -> dict[str, Any] | None:
    """Load OCR json block file for a given page."""
    json_file = book_dir / "raw" / "json" / f"page_{page_num:04d}.json"
    if not json_file.exists():
        return None
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"Failed to load OCR JSON {json_file}: {e}")
        return None


def get_book_title_from_ocr_json(ocr_json: dict[str, Any] | None) -> str | None:
    """Extract book title candidate from SectionHeader blocks in OCR JSON."""
    if not ocr_json:
        return None
    blocks = ocr_json.get("blocks", [])
    for block in blocks:
        block_type = block.get("type", "")
        block_label = block.get("label", "")
        if block_type == "SectionHeader" or block_label in ["Section-title", "Header"]:
            text = (block.get("text") or "").strip()
            # If text is not just a digit or very short symbol
            if len(text) > 2 and not text.isdigit():
                return text
    return None


def get_page_list(book_dir: Path) -> list[int]:
    """Get sorted list of page numbers available in a book's raw markdown directory."""
    md_dir = book_dir / "raw" / "markdown"
    if not md_dir.exists():
        return []
    page_numbers: list[int] = []
    for p in md_dir.glob("page_*.md"):
        m = re.match(r"page_(\d+)\.md", p.name)
        if m:
            page_numbers.append(int(m.group(1)))
    return sorted(page_numbers)


def parse_pages_arg(pages_str: str) -> list[int]:
    """
    Parse page argument like '50', '1-50', '1,2,5-10'.
    """
    pages: set[int] = set()
    for part in pages_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            for p in range(start, end + 1):
                pages.add(p)
        else:
            pages.add(int(part))
    return sorted(pages)


def call_agy(
    prompt: str,
    schema_path: Path,
    model: str | None = None,
    effort: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """
    Call Antigravity CLI via subprocess with --json-schema and piped stdin prompt.
    """
    cmd = [
        "agy",
        "--disable-slash-commands",
        "--dangerously-skip-permissions",
        "--json-schema",
        str(schema_path.resolve()),
        "--output-format",
        "json",
        "-p",
        prompt,
    ]
    if model:
        cmd.extend(["--model", model])
    if effort:
        # Note: if model name already ends in -low/-medium/-high etc., agy rejects --effort
        if not (model and any(model.endswith(f"-{s}") for s in ["low", "medium", "high", "xhigh", "max"])):
            cmd.extend(["--effort", effort])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        err_msg = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"agy exited with code {result.returncode}: {err_msg}")

    stdout_text = result.stdout.strip()
    if not stdout_text:
        raise RuntimeError("agy returned empty output")

    try:
        data = json.loads(stdout_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"agy output is not valid JSON: {stdout_text[:200]} ({e})")

    if data.get("status") == "ERROR":
        raise RuntimeError(f"agy session error: {data.get('error')}")

    # Check for direct structured_output from agy
    if "structured_output" in data and isinstance(data["structured_output"], dict):
        return data["structured_output"]

    # Fallback to parsing response field
    response_str = data.get("response", "")
    if isinstance(response_str, dict):
        return response_str

    if response_str:
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            # Try finding json substring in response
            m = re.search(r"\{.*\}", response_str, re.DOTALL)
            if m:
                return json.loads(m.group(0))

    raise RuntimeError(f"Could not extract structured JSON from agy response: {data}")


def call_openai_api(
    prompt: str,
    schema: dict[str, Any],
    base_url: str = "http://localhost:8080",
    model: str = "local-model",
    timeout: int = 120,
) -> dict[str, Any]:
    """
    Call OpenAI-compatible endpoint at base_url using standard urllib.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        resp_json = json.loads(body)
        content = resp_json["choices"][0]["message"]["content"]
        return json.loads(content)


def validate_and_clean_result(
    result: dict[str, Any],
    raw_page_text: str,
    default_book_name_gujarati: str,
    default_book_name_english: str,
    default_book_page_no: str,
) -> dict[str, Any]:
    """
    Ensure all fields required by page_metadata_schema.json schema exist and are formatted properly.
    """
    raw_book_name = result.get("book_name")
    if isinstance(raw_book_name, dict):
        guj_name = str(raw_book_name.get("gujarati") or default_book_name_gujarati).strip()
        eng_name = str(raw_book_name.get("english") or default_book_name_english).strip()
    elif isinstance(raw_book_name, str) and raw_book_name.strip():
        guj_name = raw_book_name.strip()
        eng_name = default_book_name_english
    else:
        guj_name = default_book_name_gujarati
        eng_name = default_book_name_english

    book_name = {
        "gujarati": guj_name,
        "english": eng_name,
    }

    # book_page_no: file sequence number
    raw_book_page_no = result.get("book_page_no")
    if raw_book_page_no is not None and str(raw_book_page_no).strip():
        book_page_no = str(raw_book_page_no).strip()
    else:
        book_page_no = str(default_book_page_no).strip()

    # content_page_no: page number found in content
    raw_content_page_no = result.get("content_page_no")
    if raw_content_page_no is None:
        raw_content_page_no = result.get("page_no", "")
    content_page_no = str(raw_content_page_no).strip() if raw_content_page_no is not None else ""

    def clean_list(val: Any) -> list[str]:
        if not isinstance(val, list):
            if isinstance(val, str) and val.strip():
                return [val.strip()]
            return []
        items = []
        for x in val:
            if x is not None:
                s = str(x).strip()
                if s and s not in items:
                    items.append(s)
        return items

    persons = clean_list(result.get("persons_mentioned"))
    places = clean_list(result.get("places_mentioned"))
    dates = clean_list(result.get("dates_mentioned"))
    festivals = clean_list(result.get("festivals_events_mentioned"))

    # Post-clean text content
    text_content = result.get("text_content")
    if not text_content or not str(text_content).strip():
        text_content = clean_text_content(raw_page_text)
    else:
        text_content = clean_text_content(str(text_content))

    return {
        "book_name": book_name,
        "book_page_no": book_page_no,
        "content_page_no": content_page_no,
        "persons_mentioned": persons,
        "places_mentioned": places,
        "dates_mentioned": dates,
        "festivals_events_mentioned": festivals,
        "text_content": text_content,
    }


def process_page(
    book_dir: Path,
    book_folder_name: str,
    page_num: int,
    output_dir: Path,
    schema_path: Path,
    schema_dict: dict[str, Any],
    config: dict[str, Any],
    tracker: ProgressTracker,
    total_pages: int,
) -> bool:
    """
    Process a single page: load -> prompt -> call LLM (with retries) -> validate -> save.
    """
    out_book_dir = output_dir / book_folder_name
    out_file = out_book_dir / f"page_{page_num:04d}.json"

    if config.get("resume") and tracker.is_processed(book_folder_name, page_num) and out_file.exists():
        logger.info(f"[{book_folder_name} p.{page_num}] Already processed, skipping.")
        return True

    raw_text = load_page_text(book_dir, page_num)
    if not raw_text or not raw_text.strip():
        logger.warning(f"[{book_folder_name} p.{page_num}] Empty or missing markdown file.")
        tracker.mark_failed(book_folder_name, page_num, total_pages)
        return False

    ocr_json = load_ocr_json(book_dir, page_num)
    title_hint = get_book_title_from_ocr_json(ocr_json)

    prompt = build_prompt(
        page_text=raw_text,
        book_folder_name=book_folder_name,
        book_title_hint=title_hint,
        file_page_no=str(page_num),
    )

    max_retries = 3
    result_dict: dict[str, Any] | None = None
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            if config.get("use_openai"):
                result_dict = call_openai_api(
                    prompt=prompt,
                    schema=schema_dict,
                    base_url=config.get("openai_url", "http://localhost:8080"),
                    model=config.get("model", "local-model"),
                )
            else:
                result_dict = call_agy(
                    prompt=prompt,
                    schema_path=schema_path,
                    model=config.get("model"),
                    effort=config.get("effort"),
                )
            if result_dict:
                break
        except Exception as e:
            last_error = e
            wait_time = attempt * 2
            logger.warning(
                f"[{book_folder_name} p.{page_num}] Attempt {attempt}/{max_retries} failed: {e}. "
                f"Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)

    if not result_dict:
        logger.error(f"[{book_folder_name} p.{page_num}] Failed after {max_retries} attempts: {last_error}")
        # Save error report in _errors directory
        err_dir = output_dir / "_errors" / book_folder_name
        err_dir.mkdir(parents=True, exist_ok=True)
        err_file = err_dir / f"page_{page_num:04d}.error.txt"
        err_file.write_text(f"Error: {last_error}\nPrompt:\n{prompt}", encoding="utf-8")
        tracker.mark_failed(book_folder_name, page_num, total_pages)
        return False

    cleaned_data = validate_and_clean_result(
        result=result_dict,
        raw_page_text=raw_text,
        default_book_name_gujarati=title_hint or book_folder_name,
        default_book_name_english=book_folder_name,
        default_book_page_no=str(page_num),
    )

    out_book_dir.mkdir(parents=True, exist_ok=True)
    temp_out = out_file.with_suffix(".tmp")
    with open(temp_out, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
    temp_out.replace(out_file)

    tracker.mark_processed(book_folder_name, page_num, total_pages)
    logger.info(
        f"[{book_folder_name} p.{page_num}] Successfully extracted: "
        f"{len(cleaned_data['persons_mentioned'])} persons, "
        f"{len(cleaned_data['places_mentioned'])} places, "
        f"{len(cleaned_data['dates_mentioned'])} dates, "
        f"{len(cleaned_data['festivals_events_mentioned'])} events"
    )
    return True


def process_book(
    book_folder_name: str,
    data_dir: Path,
    output_dir: Path,
    schema_path: Path,
    schema_dict: dict[str, Any],
    config: dict[str, Any],
    tracker: ProgressTracker,
    selected_pages: list[int] | None = None,
) -> dict[str, int]:
    """Process all or selected pages for a single book."""
    book_dir = data_dir / book_folder_name
    if not book_dir.exists():
        logger.error(f"Book directory {book_dir} does not exist.")
        return {"total": 0, "success": 0, "failed": 0}

    all_pages = get_page_list(book_dir)
    total_book_pages = len(all_pages)

    if selected_pages:
        pages_to_process = [p for p in selected_pages if p in all_pages]
        if not pages_to_process:
            logger.warning(
                f"None of requested pages {selected_pages} found in book {book_folder_name}. "
                f"Available pages: 1 to {total_book_pages}."
            )
            return {"total": 0, "success": 0, "failed": 0}
    else:
        pages_to_process = all_pages

    logger.info(
        f"Processing book '{book_folder_name}': {len(pages_to_process)} pages to run "
        f"(out of {total_book_pages} total)."
    )

    if config.get("dry_run"):
        print(f"\n[DRY RUN] Book: {book_folder_name}")
        print(f"  Total book pages: {total_book_pages}")
        print(f"  Pages to process ({len(pages_to_process)}): {pages_to_process[:20]}{'...' if len(pages_to_process) > 20 else ''}")
        return {"total": len(pages_to_process), "success": 0, "failed": 0}

    concurrency = config.get("concurrency", 1)
    success_count = 0
    failed_count = 0

    if concurrency > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_page = {
                executor.submit(
                    process_page,
                    book_dir=book_dir,
                    book_folder_name=book_folder_name,
                    page_num=p,
                    output_dir=output_dir,
                    schema_path=schema_path,
                    schema_dict=schema_dict,
                    config=config,
                    tracker=tracker,
                    total_pages=total_book_pages,
                ): p
                for p in pages_to_process
            }
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    ok = future.result()
                    if ok:
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as exc:
                    logger.error(f"Page {page_num} generated unhandled exception: {exc}")
                    failed_count += 1
    else:
        for p in pages_to_process:
            ok = process_page(
                book_dir=book_dir,
                book_folder_name=book_folder_name,
                page_num=p,
                output_dir=output_dir,
                schema_path=schema_path,
                schema_dict=schema_dict,
                config=config,
                tracker=tracker,
                total_pages=total_book_pages,
            )
            if ok:
                success_count += 1
            else:
                failed_count += 1

    return {"total": len(pages_to_process), "success": success_count, "failed": failed_count}


def get_all_books(data_dir: Path) -> list[str]:
    """List all available book directory names inside data_dir."""
    if not data_dir.exists():
        return []
    books = []
    for item in sorted(data_dir.iterdir()):
        if item.is_dir() and (item / "raw" / "markdown").exists():
            books.append(item.name)
    return books


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-powered metadata extraction for Gujarati scripture OCR pages.\n\n"
            "Reads OCR markdown/JSON pages, builds prompts with contextual hints,\n"
            "calls the Antigravity CLI (agy) or a local OpenAI-compatible endpoint,\n"
            "and saves one structured JSON file per page.\n\n"
            "Must specify either --book <folder> or --all."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # single book, all pages\n"
            "  python scripts/extract_metadata.py --book 72181_Shri_Harikrushna_Charitramrut_Sagar_Bhag_1\n\n"
            "  # specific page range\n"
            "  python scripts/extract_metadata.py --book <book_folder> --pages 1-50\n"
            "  python scripts/extract_metadata.py --book <book_folder> --pages 1,5,10-20\n\n"
            "  # resume a previous run (skip already-done pages)\n"
            "  python scripts/extract_metadata.py --book <book_folder> --resume\n\n"
            "  # process all books in data/ocr_output/\n"
            "  python scripts/extract_metadata.py --all --resume --concurrency 4\n\n"
            "  # use a local OpenAI-compatible server instead of agy\n"
            "  python scripts/extract_metadata.py --book <book_folder> \\\n"
            "      --use-openai --openai-url http://localhost:8080\n\n"
            "  # dry run — preview what would be processed without calling the LLM\n"
            "  python scripts/extract_metadata.py --book <book_folder> --dry-run\n\n"
            "output layout:\n"
            "  data/extracted_metadata/\n"
            "    <book_folder>/\n"
            "      page_0001.json        one structured JSON file per page\n"
            "      page_0002.json\n"
            "      ...\n"
            "    _progress.json          tracks processed/failed pages per book\n"
            "    _summary.json           aggregate summary across all books\n"
            "    _errors/<book>/         error logs for pages that failed all retries\n"
        ),
    )
    parser.add_argument("--book", type=str, default=None, help="Process a single book by folder name")
    parser.add_argument("--all", action="store_true", help="Process all books")
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Page range like '1-50' or '50' or '1,2,5-10'. Only applies when --book is specified.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ocr_output",
        help="Directory containing OCR output books (default: data/ocr_output)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/extracted_metadata",
        help="Where to save output JSONs (default: data/extracted_metadata)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="scripts/page_metadata_schema.json",
        help="Path to JSON Schema file (default: scripts/page_metadata_schema.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.8-flash-low",
        help="Override LLM model (default: gemini-3.8-flash-low)",
    )
    parser.add_argument(
        "--effort",
        type=str,
        default=None,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort for agy (low/medium/high)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of parallel invocations (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip pages that already have output JSONs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed without calling LLM",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI-compatible API at localhost:8080 instead of agy",
    )
    parser.add_argument(
        "--openai-url",
        type=str,
        default="http://localhost:8080",
        help="OpenAI-compatible API base URL (default: http://localhost:8080)",
    )

    args = parser.parse_args()

    if not args.book and not args.all:
        parser.error("Must specify either --book <name> or --all")

    if args.pages and args.all:
        parser.error("--pages can only be used with --book, not with --all")

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    schema_path = Path(args.schema).resolve()

    if not schema_path.exists():
        logger.error(f"Schema file not found at: {schema_path}")
        return 1

    with open(schema_path, "r", encoding="utf-8") as f:
        schema_dict = json.load(f)

    tracker = ProgressTracker(output_dir)

    config = {
        "model": args.model,
        "effort": args.effort,
        "concurrency": args.concurrency,
        "resume": args.resume,
        "dry_run": args.dry_run,
        "use_openai": args.use_openai,
        "openai_url": args.openai_url,
    }

    selected_pages = parse_pages_arg(args.pages) if args.pages else None

    if args.book:
        target_books = [args.book]
    else:
        target_books = get_all_books(data_dir)
        if not target_books:
            logger.error(f"No books with raw/markdown found in {data_dir}")
            return 1
        logger.info(f"Found {len(target_books)} books to process.")

    total_stats = {"books": 0, "total_pages": 0, "success": 0, "failed": 0}

    for b in target_books:
        stats = process_book(
            book_folder_name=b,
            data_dir=data_dir,
            output_dir=output_dir,
            schema_path=schema_path,
            schema_dict=schema_dict,
            config=config,
            tracker=tracker,
            selected_pages=selected_pages,
        )
        total_stats["books"] += 1
        total_stats["total_pages"] += stats["total"]
        total_stats["success"] += stats["success"]
        total_stats["failed"] += stats["failed"]

    tracker.save_summary()

    logger.info("=" * 60)
    logger.info("Extraction Run Completed")
    logger.info(
        f"Books: {total_stats['books']} | "
        f"Pages: {total_stats['total_pages']} | "
        f"Success: {total_stats['success']} | "
        f"Failed: {total_stats['failed']}"
    )
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("=" * 60)

    return 0 if total_stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
