# How to Use the Vector Database & Query Pipeline

This guide explains how to query, filter, and run RAG (Retrieval-Augmented Generation) on the Qdrant vector database created for Gujarati scriptures.

---

## 1. Environment & Setup

Ensure the virtual environment is activated and required dependencies are installed:

```bash
cd /home/yogesh/Documents/repos/find-book-reference
source .venv/bin/activate
```

Core dependencies:
- `qdrant-client` — Vector database client (supports local embedded storage and remote servers)
- `sentence-transformers` — Generates dense embeddings (`intfloat/multilingual-e5-large`)
- `torch` — PyTorch backend for tensor embeddings

---

## 2. CLI Tool: `scripts/t4_query.py` (Recommended)

`scripts/t4_query.py` is the primary, production-ready interface for querying the vector database. It supports semantic search, metadata filtering, interactive shell queries, and full RAG answer synthesis with exact source citations.

### A. Quick Semantic Search (Search-Only)

Run a fast vector retrieval query from the command line without calling an LLM:

```bash
python scripts/t4_query.py -q "માધવદાસ સ્વામી ક્યાં ગયા?" --no-llm
```

Retrieve top 3 results with a custom confidence threshold:

```bash
python scripts/t4_query.py -q "મહારાજ અને સંતો" --top-k 3 --score-threshold 0.40 --no-llm
```

### B. Filtered Search (Book, Person, Place)

Narrow down search results using structured metadata fields:

```bash
# Filter by Place
python scripts/t4_query.py -q "પાતળભાઈની વાત" --place "ભાડેર" --no-llm

# Filter by Person
python scripts/t4_query.py -q "દર્શન આપ્યા" --person "માધવદાસ સ્વામી" --no-llm

# Filter by Book Name (English or Gujarati title)
python scripts/t4_query.py -q "ચમત્કાર" --book "Shri Harikrushna Charitramrut Sagar Bhag 1" --no-llm

# Combine multiple filters
python scripts/t4_query.py -q "સભા" --book "Shri Harikrushna Charitramrut Sagar Bhag 1" --place "ગઢડા" --no-llm
```

### C. Interactive REPL Mode (Fastest for Multiple Queries)

In interactive mode, the embedding model (`multilingual-e5-large`) loads **once into memory**, allowing instant subsequent queries without reload overhead:

```bash
python scripts/t4_query.py --interactive
```

*(You can also use `-i` as a shortcut)*:
```bash
python scripts/t4_query.py -i --no-llm
```
Type your query and press `Enter`. Type `exit` or `q` to quit.

### D. Full RAG Q&A (Retrieve Context + Synthesize Answer)

`t4_query.py` retrieves the most relevant scripture passages, builds an authoritative prompt with exact citations, and requests an answer from an LLM.

#### 1. Local OpenAI-Compatible Server (llama-server, vLLM, Ollama)
```bash
python scripts/t4_query.py -q "ભાડેરના પાતળભાઈનો પ્રસંગ શું છે?" \
    --backend openai \
    --openai-url http://localhost:8080/v1
```

#### 2. Antigravity CLI (`agy`)
```bash
python scripts/t4_query.py -q "ભાડેરના પાતળભાઈનો પ્રસંગ શું છે?" \
    --backend agy
```

### E. JSON Output for Automation & APIs

Output structured JSON (suitable for piping into `jq`, web backends, or automated pipelines):

```bash
python scripts/t4_query.py -q "ભાડેર ગામ" --no-llm --json
```

---

## 3. `t4_query.py` CLI Options Reference

| Argument | Shorthand | Default | Description |
|---|---|---|---|
| `--query` | `-q` | `None` | Query string (Gujarati or English). Launches interactive mode if omitted. |
| `--interactive` | `-i` | `False` | Run interactive REPL shell (keeps model in memory). |
| `--top-k` | `-k` | `5` | Maximum number of chunks to retrieve. |
| `--score-threshold` | | `0.35` | Minimum cosine similarity score (0.0 to 1.0). |
| `--max-context-chunks` | | `5` | Maximum chunks forwarded to the LLM context. |
| `--book` | | `None` | Filter chunks by book title (English or Gujarati). |
| `--person` | | `None` | Filter chunks mentioning this specific person. |
| `--place` | | `None` | Filter chunks mentioning this specific place/village. |
| `--no-llm` / `--search-only` | | `False` | Run vector search only; do not call LLM. |
| `--backend` | | `openai` | LLM backend: `openai` or `agy`. |
| `--openai-url` | | `http://localhost:8080/v1` | Base URL for OpenAI-compatible LLM endpoint. |
| `--model` | | `None` | Model name identifier passed to the LLM backend. |
| `--temperature` | | `0.1` | LLM sampling temperature. |
| `--max-tokens` | | `2048` | Maximum tokens to generate. |
| `--qdrant-path` | | `./data/vector_db` | Path to embedded Qdrant database directory. |
| `--qdrant-host` | | `None` | Remote Qdrant server hostname/IP. |
| `--qdrant-port` | | `6333` | Remote Qdrant server port. |
| `--collection` | | `gujarati_scriptures` | Qdrant collection name. |
| `--embed-model` | | `intfloat/multilingual-e5-large` | SentenceTransformer model identifier. |
| `--device` | | `None` | PyTorch device (`cpu`, `cuda`). |
| `--json` | | `False` | Output results as JSON. |
| `--no-color` | | `False` | Disable ANSI terminal colors. |

---

## 4. Programmatic Python API

You can integrate scripture search into custom Python scripts in two ways:

### Method A: Using `ScriptureQueryEngine` (High-Level, Recommended)

Reuse the existing query engine directly without writing boilerplate code:

```python
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path("./scripts").resolve()))
from t4_query import ScriptureQueryEngine

# 1. Initialize engine
engine = ScriptureQueryEngine(
    qdrant_path="./data/vector_db",
    collection_name="gujarati_scriptures",
)

# 2. Perform search with optional filters
results = engine.search(
    query="માધવદાસ સ્વામી ક્યાં ગયા?",
    top_k=5,
    score_threshold=0.35,
    place_filter="ભાડેર",  # Optional
)

# 3. Process results
for hit in results:
    print(f"--- [Score: {hit['score']:.4f}] ---")
    print(f"Book   : {hit['book_name'].get('gujarati')} ({hit['book_name'].get('english')})")
    print(f"Pages  : Book Page {hit['book_page_no']}, Content Page {hit['content_page_no']}")
    print(f"Persons: {hit['persons_mentioned']}")
    print(f"Places : {hit['places_mentioned']}")
    print(f"Text   :\n{hit['text_content'][:200]}...\n")
```

### Method B: Direct `qdrant-client` & `sentence-transformers` (Low-Level)

> [!NOTE]
> For `intfloat/multilingual-e5-large`, passages were indexed with prefix `"passage: "`. When querying, **always prefix questions with `"query: "`**.

```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 1. Connect to local embedded Qdrant storage
client = QdrantClient(path="./data/vector_db")

# 2. Load the embedding model
embed_model = SentenceTransformer("intfloat/multilingual-e5-large")

def search(question: str, top_k: int = 5):
    # Prefix query with 'query: ' for e5 models
    query_vec = embed_model.encode(f"query: {question}", normalize_embeddings=True)

    # Query Qdrant collection
    results = client.query_points(
        collection_name="gujarati_scriptures",
        query=query_vec.tolist(),
        limit=top_k,
        score_threshold=0.35,
    )
    return results.points

points = search("માધવદાસ સ્વામી ક્યાં ગયા?")
for hit in points:
    payload = hit.payload
    print(f"Score: {hit.score:.3f} | Book: {payload['book_name']['english']} | Page: {payload['book_page_no']}")
```

---

## 5. Metadata Filtering (Low-Level Syntax)

When writing custom queries using `qdrant-client`, combine vector similarity with structured metadata filters:

### A. Filter by Book Name

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

filter_by_book = Filter(
    must=[
        FieldCondition(
            key="book_name.english",
            match=MatchValue(value="Shri Harikrushna Charitramrut Sagar Bhag 1"),
        )
    ]
)

response = client.query_points(
    collection_name="gujarati_scriptures",
    query=query_vec.tolist(),
    limit=5,
    query_filter=filter_by_book,
)
```

### B. Filter by Person Mentioned

```python
from qdrant_client.models import Filter, FieldCondition, MatchAny

filter_by_person = Filter(
    must=[
        FieldCondition(
            key="persons_mentioned",
            match=MatchAny(any=["માધવદાસ સ્વામી"]),
        )
    ]
)

response = client.query_points(
    collection_name="gujarati_scriptures",
    query=query_vec.tolist(),
    limit=5,
    query_filter=filter_by_person,
)
```

### C. Filter by Place Mentioned

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

filter_by_place = Filter(
    must=[
        FieldCondition(
            key="places_mentioned",
            match=MatchValue(value="અમદાવાદ"),
        )
    ]
)

response = client.query_points(
    collection_name="gujarati_scriptures",
    query=query_vec.tolist(),
    limit=5,
    query_filter=filter_by_place,
)
```

---

## 6. Payload Schema Reference

Each point in the `gujarati_scriptures` collection contains the following payload fields:

| Field | Type | Description |
|---|---|---|
| `book_name` | `dict` | Book name object: `{"gujarati": "...", "english": "..."}` |
| `book_page_no` | `str` | Physical scanned page sequence (e.g. `"23"` or `"23,24"` if chunk spans pages) |
| `content_page_no` | `str` | Printed page number appearing on the book page (e.g. `"143"`, `"૮૫"`) |
| `text_content` | `str` | Full chunk text (~500 words per chunk) |
| `persons_mentioned` | `list[str]` | Persons, saints, and deities mentioned in chunk |
| `places_mentioned` | `list[str]` | Cities, villages, temples, rivers mentioned in chunk |
| `dates_mentioned` | `list[str]` | Dates mentioned in chunk (e.g. `"સં. ૧૮૮૦"`) |
| `festivals_events_mentioned` | `list[str]` | Festivals, ceremonies, or events mentioned |
| `chunk_index` | `int` | Sequential chunk index within the book |

### Vector Index Parameters
- **Model**: `intfloat/multilingual-e5-large`
- **Embedding Dimensions**: 1024
- **Distance Metric**: `Cosine`
- **Prefix Requirement**: Passages indexed with `"passage: "`, search queries prefixed with `"query: "`

---

## 7. Collection Inspection & Database Maintenance

### Check Collection Status and Points Count

```python
from qdrant_client import QdrantClient

client = QdrantClient(path="./data/vector_db")
info = client.get_collection("gujarati_scriptures")

print("Collection Status :", info.status)
print("Points Count      :", info.points_count)
print("Vector Size       :", info.config.params.vectors.size)
print("Distance Metric   :", info.config.params.vectors.distance)
```

### Indexing New Books / Re-indexing

To index newly extracted books or rebuild the vector database, use `scripts/t3_index_data.py`:

```bash
# Index all books found in data/extracted_metadata/
python scripts/t3_index_data.py

# Index a single book only
python scripts/t3_index_data.py --book 72181_Shri_Harikrushna_Charitramrut_Sagar_Bhag_1

# Re-index from scratch (drops and recreates the collection)
python scripts/t3_index_data.py --reindex
```
