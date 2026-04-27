# Pharma Document RAG (Showcase Version)

AI-powered question answering over pharmaceutical PDF documents with source-grounded responses.

## Demo Video

- Primary demo (Loom): [Watch demo on Loom](https://www.loom.com/share/7e9392e6ba2049299db590e544a34cd0)
- Local backup: [`demo/RAG Pipeline for PDF Document QA.mp4`](demo/RAG%20Pipeline%20for%20PDF%20Document%20QA.mp4)

If your Markdown renderer supports HTML video tags:

<video src="demo/RAG Pipeline for PDF Document QA.mp4" controls width="900"></video>

## Why this project

Pharma teams often need fast answers from long, mixed-quality document sets (COAs, processing certificates, packaging specs, declarations, etc.).  
This system turns those PDFs into a searchable knowledge base and returns concise answers with traceable sources.

## What it does

- Ingests and parses PDFs page-by-page
- Detects low-quality text and applies OCR when needed
- Classifies pages into pharmaceutical document types
- Merges pages into logical documents
- Chunks and embeds content into FAISS indexes
- Routes queries by predicted document type
- Reranks results with a cross-encoder
- Generates source-grounded answers in a Gradio UI
- Prioritizes uploaded PDFs over the base library at query time

## Quickstart

### 1) Install

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment

Create `.env` in project root:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=models/gemini-2.5-flash

EMBEDDER_PROVIDER=bge
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

### 3) Add PDFs

Put library PDFs in:

```text
data/raw_pdfs/
```

### 4) Run app

```bash
python -m src.app.app
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860)

## Typical UI workflow

1. Click **Load Saved Index** (if existing index is available), or **Rebuild Index From PDFs**
2. Ask a question in chat
3. Review returned sources and chunk previews
4. Optionally upload PDFs and click **Index uploaded PDFs** (uploads are searched first)

## Architecture at a glance

```text
PDFs -> Parse/OCR -> Doc-type classification -> Logical document builder
     -> Chunking -> Embeddings -> FAISS indexing (global + per-doc-type)
     -> Query router -> Retrieval -> Cross-encoder rerank -> Answer generation
```

## Models and techniques

### LLM + prompting

- Gemini via LlamaIndex (`llama-index-llms-google-genai`)
- Default model: `models/gemini-2.5-flash`
- LLM retry wrappers for reliability during indexing and retrieval-time calls
- Source-aware constrained prompting for answer generation

### Ingestion and preprocessing

- PDF extraction with PyMuPDF
- OCR fallback with Tesseract (`pytesseract`) for low-text/poor-text pages
- OCR trigger heuristics based on text length, word count, and unusual character ratio
- Text cleaning before classification/chunking

### Document understanding

- Pharma-specific page classification into document types:
  - `cover_letter`
  - `certificate_of_quality`
  - `packaging_specification`
  - `bse_tse_declaration`
  - `material_description`
  - `supplier_qualification`
  - `chain_of_custody`
  - `certificate_of_processing`
  - `unknown`
- Continuation detection across pages:
  - rule-based scoring (doc number patterns, sequential pages, continuation phrases)
  - optional LLM fallback for ambiguous transitions
- Logical document grouping before chunking

### Retrieval stack

- Embeddings: `BAAI/bge-base-en-v1.5` (SentenceTransformers)
- Vector store: FAISS (`IndexFlatL2`)
- Indexing strategy:
  - one global index over all chunks
  - per-doc-type indices for routed retrieval
- Query routing by predicted doc type + confidence threshold
- Two-stage retrieval:
  - routed search first
  - global fallback if needed
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Score-threshold filtering before generation

### Index persistence

- Saved index path: `storage/pharma_index`
- Persisted files:
  - `index.faiss`
  - `chunks.pkl`
- Upload index is temporary and separate from the library index

## Config and defaults

`configs/config.yaml` currently contains:

- `embedding_model`: `sentence-transformers/all-MiniLM-L6-v2`
- `reranker_model`: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `top_k`: `5`
- `top_n`: `2`

Code defaults (`src/utils/config.py`):

- `LLM_PROVIDER=gemini`
- `GEMINI_MODEL=models/gemini-2.5-flash`
- `EMBEDDER_PROVIDER=bge`
- `EMBEDDING_MODEL=BAAI/bge-base-en-v1.5`

Optional model pre-download:

```bash
python -m src.utils.download_models
```

## Project structure

```text
.
|-- configs/
|-- demo/
|-- src/
|   |-- app/
|   |-- ingestion/
|   |-- embedding/
|   |-- indexing/
|   |-- retrieval/
|   |-- pipelines/
|   |-- services/
|   |-- utils/
|   `-- test/
|-- requirements.txt
|-- README.md
`-- README2.md
```

## Testing

```bash
pytest src/test
```

## Notes

- App runs locally on `127.0.0.1:7860` by default.
- Retrieval output includes source labels and chunk previews for traceability.
