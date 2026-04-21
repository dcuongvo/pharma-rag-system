import os
import shutil
import tempfile
from pathlib import Path

import gradio as gr

from src.pipelines.indexing_pipeline import IndexingPipeline
from src.pipelines.retrieval_pipeline import RetrievalPipeline


PDF_FOLDER = "data/raw_pdfs"
SAVE_DIR = "storage/pharma_index"

FIXED_CHUNK_SIZE = 300
FIXED_OVERLAP = 50

DEFAULT_UPLOAD_MD = (
    "No upload index. Add PDFs below and click **Index uploaded PDFs**. "
    "When an upload index exists, queries search **uploads first**, then the library."
)


class ShowcaseState:
    def __init__(self):
        self.indexing_pipeline = None
        self.retrieval_pipeline = None
        self.index_result = None
        self.ready = False
        self.temp_dir = None
        self.upload_dir = None


state = ShowcaseState()


def _pipeline_has_any_index(rp: RetrievalPipeline | None) -> bool:
    if rp is None:
        return False
    lib_ok = rp.vector_store.index is not None and bool(rp.vector_store.chunks)
    up = rp.upload_vector_store
    up_ok = up is not None and up.index is not None and bool(up.chunks)
    return lib_ok or up_ok


def reset_state():
    global state
    if state.temp_dir and os.path.exists(state.temp_dir):
        shutil.rmtree(state.temp_dir, ignore_errors=True)
    if state.upload_dir and os.path.exists(state.upload_dir):
        shutil.rmtree(state.upload_dir, ignore_errors=True)
    state = ShowcaseState()


def _clear_upload_index_only():
    global state
    if state.retrieval_pipeline is not None:
        state.retrieval_pipeline.upload_vector_store = None
    if state.upload_dir and os.path.exists(state.upload_dir):
        shutil.rmtree(state.upload_dir, ignore_errors=True)
    state.upload_dir = None
    state.ready = _pipeline_has_any_index(state.retrieval_pipeline)


def _format_structure_from_index_result(index_result):
    logical_docs = index_result.get("logical_docs", [])
    if not logical_docs:
        return "No logical documents found."

    lines = []
    for doc in logical_docs:
        lines.append(
            f"• `{doc.doc_type}` | pages {doc.page_start} to {doc.page_end} | id: `{doc.logical_doc_id}`"
        )
    return "\n".join(lines)


def _format_status_from_index_result(index_result):
    stats = index_result["stats"]
    return f"""
### Index Ready

**Pages:** {stats['num_pages']}  
**Logical documents:** {stats['num_logical_docs']}  
**Chunks:** {stats['num_chunks']}  
**Embedder:** {stats['embedder_model']}  
**Chunk size:** {stats['chunk_size']}  
**Overlap:** {stats['overlap']}
"""


def load_saved_index():
    try:
        reset_state()

        retrieval_pipeline = RetrievalPipeline(
            rerank_score_threshold=-999.0,
        )
        retrieval_pipeline.load_index(SAVE_DIR)

        state.retrieval_pipeline = retrieval_pipeline
        state.ready = True

        num_chunks = len(retrieval_pipeline.vector_store.chunks)
        doc_types = sorted(list(retrieval_pipeline.vector_store.doc_type_indices.keys()))

        status_md = f"""
### Loaded Saved Index

**Path:** `{SAVE_DIR}`  
**Chunks:** {num_chunks}  
**Document types:** {", ".join(doc_types) if doc_types else "None"}
"""
        structure_md = "Saved index loaded successfully."
        return (
            status_md,
            structure_md,
            [],
            "No sources yet.",
            "No supporting chunks yet.",
            "",
            DEFAULT_UPLOAD_MD,
        )

    except Exception as e:
        return (
            f"### Error\n`{str(e)}`",
            "",
            [],
            "No sources yet.",
            "No supporting chunks yet.",
            "",
            DEFAULT_UPLOAD_MD,
        )


def rebuild_index():
    try:
        reset_state()

        state.indexing_pipeline = IndexingPipeline(
            chunk_size=FIXED_CHUNK_SIZE,
            overlap=FIXED_OVERLAP,
        )

        index_result = state.indexing_pipeline.run(PDF_FOLDER)
        state.index_result = index_result

        index_result["vector_store"].save(SAVE_DIR)

        state.retrieval_pipeline = RetrievalPipeline(
            vector_store=index_result["vector_store"],
            rerank_score_threshold=-999.0,
        )

        state.ready = True

        status_md = _format_status_from_index_result(index_result) + f"\n\n**Saved to:** `{SAVE_DIR}`"
        structure_md = _format_structure_from_index_result(index_result)

        return (
            status_md,
            structure_md,
            [],
            "No sources yet.",
            "No supporting chunks yet.",
            "",
            DEFAULT_UPLOAD_MD,
        )

    except Exception as e:
        return (
            f"### Error\n`{str(e)}`",
            "",
            [],
            "No sources yet.",
            "No supporting chunks yet.",
            "",
            DEFAULT_UPLOAD_MD,
        )


def index_uploaded_pdfs(files):
    global state
    try:
        if files is None:
            file_list = []
        elif isinstance(files, list):
            file_list = [f for f in files if f]
        else:
            file_list = [files]

        if not file_list:
            _clear_upload_index_only()
            return DEFAULT_UPLOAD_MD

        state.upload_dir = state.upload_dir or tempfile.mkdtemp(prefix="pharma_upload_")
        upload_path = Path(state.upload_dir)
        for p in upload_path.glob("*.pdf"):
            p.unlink()
        for f in file_list:
            shutil.copy2(f, state.upload_dir)

        state.indexing_pipeline = IndexingPipeline(
            chunk_size=FIXED_CHUNK_SIZE,
            overlap=FIXED_OVERLAP,
        )
        index_result = state.indexing_pipeline.run(str(upload_path))

        if state.retrieval_pipeline is None:
            state.retrieval_pipeline = RetrievalPipeline(
                rerank_score_threshold=-999.0,
            )
        state.retrieval_pipeline.upload_vector_store = index_result["vector_store"]

        state.ready = _pipeline_has_any_index(state.retrieval_pipeline)

        n = len(index_result["chunks"])
        return (
            f"**Upload index:** {n} chunks (temp: `{state.upload_dir}`). "
            f"Queries use **uploads first**, then the library index if loaded."
        )

    except Exception as e:
        return f"### Upload error\n`{str(e)}`"


def clear_upload_index():
    _clear_upload_index_only()
    return DEFAULT_UPLOAD_MD


def ask_question(message, history, top_k, retrieve_k):
    history = history or []

    if not _pipeline_has_any_index(state.retrieval_pipeline):
        history.append({"role": "user", "content": message})
        history.append(
            {
                "role": "assistant",
                "content": "Please load or rebuild the library index, or index uploaded PDFs first.",
            }
        )
        return history, "No sources yet.", "No supporting chunks yet.", ""

    if not message or not message.strip():
        return history, "No sources yet.", "No supporting chunks yet.", ""

    try:
        result = state.retrieval_pipeline.query(
            query_text=message,
            top_k=top_k,
            retrieve_k=retrieve_k,
        )

        sources = result.get("sources", [])
        supporting_chunks = result.get("results", [])

        n_total = max(len(sources), len(supporting_chunks))
        source_lines = []
        for i, src in enumerate(sources, start=1):
            # Running index 1…N; aligns with Source N in the model context / answer.
            denom = n_total if n_total else i
            source_lines.append(
                f"{i}. **Ref {i}/{denom}** — {src.get('doc_type')} | "
                f"pages {src.get('page_start')} to {src.get('page_end')} | "
                f"score {src.get('score', 0.0):.4f}"
            )
        if source_lines:
            n = len(sources)
            if n == 1:
                intro = "*Number **1** matches **Source 1** in the reply.*\n\n"
            else:
                intro = f"*Numbers 1–{n} match **Source 1** … **Source {n}** in the reply.*\n\n"
            sources_md = intro + "\n\n".join(source_lines)
        else:
            sources_md = "No sources returned."

        chunk_lines = []
        denom = n_total if n_total else 1
        for i, (chunk, score) in enumerate(supporting_chunks, start=1):
            chunk_lines.append(
                "\n".join(
                    [
                        f"### Ref {i}/{denom} — {chunk.metadata.get('doc_type', 'unknown')}",
                        f"**Chunk ID:** {chunk.chunk_id}",
                        f"**Document:** {chunk.document_name}",
                        f"**Doc type:** {chunk.metadata.get('doc_type')}",
                        f"**Pages:** {chunk.page_start} to {chunk.page_end}",
                        f"**Rerank score:** {score:.4f}",
                        f"**Preview:** {chunk.text[:250]}...",
                    ]
                )
            )
        chunks_md = "\n\n---\n\n".join(chunk_lines) if chunk_lines else "No supporting chunks."

        assistant_msg = (
            f"{result['answer']}\n\n"
            f"**Retrieval scope:** {result.get('retrieval_scope', 'n/a')}  \n"
            f"**Predicted doc type:** {result['predicted_doc_type']}  \n"
            f"**Used scope:** {result['used_doc_type']}  \n"
            f"**Router confidence:** {float(result.get('confidence', 0.0) or 0.0):.2f}  \n"
            f"**Answer confidence:** {float(result.get('answer_confidence', 0.0) or 0.0):.4f}"
        )

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": assistant_msg})

        return history, sources_md, chunks_md, ""

    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history, "No sources yet.", "No supporting chunks yet.", ""


def clear_all():
    reset_state()
    return (
        "Waiting for index...",
        "",
        [],
        "No sources yet.",
        "No supporting chunks yet.",
        "",
        DEFAULT_UPLOAD_MD,
    )


def build_demo():
    with gr.Blocks(title="Pharmaceutical Intelligence Document") as demo:
        gr.Markdown(
            """
# Pharmaceutical Document Q&A Showcase

This demo uses a saved FAISS index for fast loading, with an optional rebuild from the source PDFs.
Uploaded PDFs are indexed **separately** (temp folder) and searched **before** the library.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                load_btn = gr.Button("Load Saved Index", variant="primary")
                rebuild_btn = gr.Button("Rebuild Index From PDFs")
                clear_btn = gr.Button("Clear")

                status_md = gr.Markdown("Waiting for index...")
                structure_md = gr.Markdown("")

                gr.Markdown(
                    f"""
### Fixed Index Settings

**PDF folder:** `{PDF_FOLDER}`  
**Save path:** `{SAVE_DIR}`  
**Chunk size:** {FIXED_CHUNK_SIZE}  
**Overlap:** {FIXED_OVERLAP}
"""
                )

                gr.Markdown("### Uploads (prioritized)")
                upload_files = gr.File(
                    label="PDF files",
                    file_count="multiple",
                    file_types=[".pdf"],
                )
                with gr.Row():
                    index_upload_btn = gr.Button("Index uploaded PDFs", variant="primary")
                    clear_upload_btn = gr.Button("Clear upload index")
                upload_md = gr.Markdown(DEFAULT_UPLOAD_MD)

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                )

                query_box = gr.Textbox(
                    label="Ask a question",
                    placeholder="What is the lot number?",
                )

                with gr.Row():
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Final chunks",
                    )
                    retrieve_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Initial candidates",
                    )

                with gr.Row():
                    ask_btn = gr.Button("Ask", variant="primary")
                    ex1_btn = gr.Button("What is the lot number?")
                    ex2_btn = gr.Button("What is the expiration date?")
                    ex3_btn = gr.Button("Show me any date")

            with gr.Column(scale=1):
                sources_md = gr.Markdown("No sources yet.")
                chunks_md = gr.Markdown("No supporting chunks yet.")

        load_btn.click(
            fn=load_saved_index,
            outputs=[
                status_md,
                structure_md,
                chatbot,
                sources_md,
                chunks_md,
                query_box,
                upload_md,
            ],
        )

        rebuild_btn.click(
            fn=rebuild_index,
            outputs=[
                status_md,
                structure_md,
                chatbot,
                sources_md,
                chunks_md,
                query_box,
                upload_md,
            ],
        )

        index_upload_btn.click(
            fn=index_uploaded_pdfs,
            inputs=[upload_files],
            outputs=[upload_md],
        )

        clear_upload_btn.click(
            fn=clear_upload_index,
            outputs=[upload_md],
        )

        ask_btn.click(
            fn=ask_question,
            inputs=[query_box, chatbot, top_k, retrieve_k],
            outputs=[chatbot, sources_md, chunks_md, query_box],
        )

        query_box.submit(
            fn=ask_question,
            inputs=[query_box, chatbot, top_k, retrieve_k],
            outputs=[chatbot, sources_md, chunks_md, query_box],
        )

        ex1_btn.click(
            lambda: "What is the lot number?",
            outputs=[query_box],
        )

        ex2_btn.click(
            lambda: "What is the expiration date?",
            outputs=[query_box],
        )

        ex3_btn.click(
            lambda: "Show me any date",
            outputs=[query_box],
        )

        clear_btn.click(
            fn=clear_all,
            outputs=[
                status_md,
                structure_md,
                chatbot,
                sources_md,
                chunks_md,
                query_box,
                upload_md,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, debug=True)