
from src.ingestion.doc_type_classifier import DocTypeClassifier
from src.ingestion.continuation_detector import ContinuationDetector

from src.ingestion.pdf_loader import load_pdf_pages

pdf_path = "data/raw_pdfs/pharma-blob-sample.pdf"
pages = load_pdf_pages(pdf_path)

print("Classifying doc types...", flush=True)
classifier = DocTypeClassifier()
pages = classifier.attach_doc_type_to_pages(pages)

detector = ContinuationDetector()

print(f"\nTotal pages: {len(pages)}", flush=True)

for i in range(len(pages) - 1):
    prev_page = pages[i]
    curr_page = pages[i + 1]

    # only compare adjacent pages from the same source PDF
    if prev_page.document_name != curr_page.document_name:
        continue

    score = detector.continuation_score(prev_page, curr_page)
    decision = detector.should_continue(prev_page, curr_page)

    print("\n" + "=" * 70, flush=True)
    print(f"Previous: {prev_page.document_name} | page {prev_page.page_number} | {prev_page.doc_type}", flush=True)
    print(f"Current : {curr_page.document_name} | page {curr_page.page_number} | {curr_page.doc_type}", flush=True)
    print(f"Score   : {score}", flush=True)
    print(f"Continue: {decision}", flush=True)
    print(f"Prev doc number: {detector.extract_doc_number(prev_page.cleaned_text)}", flush=True)
    print(f"Curr doc number: {detector.extract_doc_number(curr_page.cleaned_text)}", flush=True)
    print(f"Curr has continuation signal: {detector.has_continuation_signal(curr_page.cleaned_text)}", flush=True)
    print(f"Prev preview: {prev_page.cleaned_text[-180:]}", flush=True)
    print(f"Curr preview: {curr_page.cleaned_text[:180]}", flush=True)