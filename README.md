# Pharma Document RAG

A retrieval-augmented generation (RAG) application for pharmaceutical PDF documents.  
The system indexes source PDFs, classifies/query-routes by document type, reranks candidates, and serves answers in a Gradio UI with source references.

## Demo Video

- Primary demo (Loom): [Watch demo on Loom](https://www.loom.com/share/7e9392e6ba2049299db590e544a34cd0)
- Local backup: [`demo/RAG Pipeline for PDF Document QA.mp4`](demo/RAG%20Pipeline%20for%20PDF%20Document%20QA.mp4)

If your Markdown renderer supports HTML video tags:

<video src="demo/RAG Pipeline for PDF Document QA.mp4" controls width="900"></video>

## Features

- PDF ingestion pipeline with logical document construction
- OCR support for scanned PDFs
- Chunking + embedding + FAISS vector indexing
- Query routing by predicted document type with confidence threshold
- Cross-encoder reranking before answer generation
- Gradio app with:
  - saved-index loading
  - full index rebuild
  - upload-priority retrieval (uploaded docs searched before library docs)
  - source and supporting-chunk transparency

## Tech Stack

- Python
- LlamaIndex
- FAISS (`faiss-cpu`)
- SentenceTransformers
- Gradio
- PyMuPDF + Tesseract OCR
- Gemini (via `llama-index-llms-google-genai`)

## Models and Techniques (Full Pipeline)

### LLM and generation

- **Provider abstraction:** `get_llm()` supports pluggable providers (current implementation uses Gemini)
- **Default LLM provider/model:** `gemini` + `models/gemini-2.5-flash`
- **Safe retry wrappers:** exponential/fixed-delay retry for LLM calls (`safe_llm_call`, `safe_indexing_llm_call`)
- **Prompted answer generation:** constrained to provided chunk context with source-aware formatting

### Ingestion and preprocessing

- **PDF parser:** PyMuPDF (`fitz`) page-level extraction
- **Adaptive OCR fallback:** Tesseract OCR (`pytesseract`) when extracted text quality is poor
- **OCR trigger heuristics:** min chars, min words, unusual-character ratio threshold
- **Text cleanup:** normalization/cleaning before downstream processing
- **Page-level document type classification:** LLM-based classifier into pharma-specific labels

### Logical document construction

- **Continuation detection heuristic scoring** using:
  - same file and consecutive pages
  - same predicted doc type
  - shared document/certificate/process IDs (regex extraction)
  - continuation phrases (e.g., "continued", "page X of")
- **LLM fallback continuation check** for ambiguous cases
- **Logical-document merge** across related pages before chunking

### Chunking and embeddings

- **Word-based chunking with overlap**
- **Configurable chunk parameters** (`chunk_size`, `overlap`)
- **Embede):** `BAAI/bge-base-en-v1.5` (SentenceTransformers via `BGEEmbedder`)
- ** codg** for reQuery instruction formattinding model (default intrieval-oriented embeddings

### Retrieval and reranking

- **Primary vector search backend:** FAISS `IndexFlatL2`
- **Dual indexing strategy:**
  - global index over all chunks
  - per-document-type FAISS sub-indices for routed retrieval
- **Query routing model:** LLM predicts doc type + confidence
- **Two-stage retrieval logic:**
  - routed retrieval by predicted doc type (when confidence is above threshold)
  - global fallback retrieval if routed path fails
- **Reranker model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (`sentence_transformers.CrossEncoder`)
- **Post-rerank score threshold** filtering before answer generation

### Retrieval scope and index strategy

- **Library index:** persistent FAISS index at `storage/pharma_index`
- **Upload index:** temporary, separate vector store built from UI-uploaded PDFs
- **Priority order at query time:** uploaded index first, then library index

### Serialization and persistence

- **FAISS index persistence:** `index.faiss`
- **Chunk metadata persistence:** `chunks.pkl`
- **Doc-type index reconstruction** on load from stored vectors/chunks

## Project Structure

```text
.
|-- configs/
|   `-- config.yaml
|-- demo/
|   `-- RAG Pipeline for PDF Document QA.mp4
|-- src/
|   |-- app/
|   |   `-- app.py
|   |-- ingestion/
|   |-- embedding/
|   |-- indexing/
|   |-- retrieval/
|   |-- pipelines/
|   |-- services/
|   |-- utils/
|   `-- test/
|-- requirements.txt
`-- README.md
```

## Prerequisites

- Python 3.10+ recommended
- Tesseract OCR installed and available in your system path (needed for scanned PDFs)

## Installation

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
# Required for current default setup
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=models/gemini-2.5-flash

# Optional overrides
EMBEDDER_PROVIDER=bge
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

## Data Layout

Place your source PDFs in:

```text
data/raw_pdfs/
```

The application saves the persistent library index to:

```text
storage/pharma_index/
```

## Running the App

From the project root:

```bash
python -m src.app.app
```

Then open:

- [http://127.0.0.1:7860](http://127.0.0.1:7860)

### Typical workflow in UI

1. Click **Load Saved Index** (if index already exists), or
2. Click **Rebuild Index From PDFs** to re-index `data/raw_pdfs`
3. Ask questions in the chat panel
4. Optionally upload PDFs and click **Index uploaded PDFs** (upload index is searched first)

## Configuration

Current config file values (see `configs/config.yaml`):

- `embedding_model`: `sentence-transformers/all-MiniLM-L6-v2`
- `reranker_model`: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `top_k`: `5`
- `top_n`: `2`

Code-level defaults from `Settings` (`src/utils/config.py`):

- `EMBEDDER_PROVIDER`: `bge`
- `EMBEDDING_MODEL`: `BAAI/bge-base-en-v1.5`
- `LLM_PROVIDER`: `gemini`
- `GEMINI_MODEL`: `models/gemini-2.5-flash`

Optional local model warmup:

```bash
python -m src.utils.download_models
```

## Testing

Run tests from project root:

```bash
pytest src/test
```

## Notes

- Retrieval runs in two stages: routed search first (when confidence is high), then global fallback.
- The app displays source references and supporting chunk previews to improve answer traceability.