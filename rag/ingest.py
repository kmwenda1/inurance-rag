# ─────────────────────────────────────────────────────────
# rag/ingest.py — Document Reading & Chunking
#
# WHAT THIS FILE DOES:
#   Takes a PDF file and turns it into small, searchable pieces.
#
# THE PROCESS:
#   PDF file → extract text per page → cut into overlapping chunks
#              → return list of chunks ready for embedding
#
# WHY WE DO IT THIS WAY:
#   An LLM can't read an entire 80-page PDF in one go — it has a
#   limit on how much text it can process at once (called a context
#   window). So we cut the document into small pieces and only send
#   the relevant ones when answering a question.
# ─────────────────────────────────────────────────────────

import os
import pdfplumber


# ── STEP 1: Extract text from a PDF ──────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Opens a PDF and extracts the text from every page.

    Returns a list of page objects, each looking like:
        {
            "page_num": 3,
            "text": "This policy covers...",
            "source": "home_policy_2024.pdf"
        }

    WHY PAGE BY PAGE?
        Because we want to know which page every piece of text came from.
        That way, when the AI gives an answer, we can say:
        "This came from page 12." That's our citation. That's how we
        prove the AI isn't making things up.

    WHY pdfplumber OVER pypdf?
        pdfplumber is better at handling tables inside PDFs.
        Insurance documents are full of premium tables, coverage tiers,
        and exclusion lists — pdfplumber extracts these more cleanly.
    """
    pages = []

    # pdfplumber opens the file and gives us access to each page
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):

            # extract_text() reads all the text on this page
            text = page.extract_text()

            # Some pages are blank or only have images — skip those
            if text and text.strip():
                pages.append({
                    "page_num": i + 1,                      # 1-indexed (page 1, not page 0)
                    "text": text.strip(),                   # remove extra whitespace
                    "source": os.path.basename(file_path)  # just the filename, not the full path
                })

    return pages


# ── STEP 2: Cut pages into overlapping chunks ─────────────────────────────────

def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 200) -> list[dict]:
    """
    Takes the list of pages and cuts each page's text into smaller chunks.

    Parameters:
        pages:      output from extract_text_from_pdf()
        chunk_size: maximum number of characters per chunk (default 800)
        overlap:    how many characters to repeat between adjacent chunks (default 200)

    Returns a list of chunk objects:
        {
            "text": "...chunk text...",
            "page_num": 3,
            "source": "home_policy_2024.pdf",
            "chunk_id": "home_policy_2024.pdf_page3_chunk0"
        }

    WHY OVERLAP?
        Imagine a critical sentence sits right at the boundary between
        two chunks — it gets split in half. Overlap means the last
        ~200 characters of Chunk A become the first ~200 characters
        of Chunk B. So that boundary sentence appears fully in both chunks.

    WHY 800 characters / 200 overlap?
        AI review of real documents showed that exclusion clauses and
        conditional exceptions ("UNLESS X", "except where Y") often sit
        just outside the 600-char boundary of the main keyword match.
        Increasing to 800/200 (25% overlap) captures these edge cases.
        Previous setting was 600/120 (20% overlap) which missed them.
    """
    chunks = []

    for page in pages:
        text = page["text"]
        page_num = page["page_num"]
        source = page["source"]

        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate where this chunk ends
            end = start + chunk_size

            # Grab the chunk text
            chunk_text = text[start:end].strip()

            # Only add non-empty chunks
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page_num": page_num,
                    "source": source,
                    # A unique ID for each chunk — needed by ChromaDB
                    "chunk_id": f"{source}_page{page_num}_chunk{chunk_index}"
                })
                chunk_index += 1

            # Move forward by (chunk_size - overlap)
            # This is how overlap works: we step back by 'overlap' characters
            # before starting the next chunk
            start = end - overlap

            # Safety: if we're near the end and overlap would make us loop
            # forever, just break
            if start >= len(text):
                break

    return chunks


# ── STEP 3: Detect if a PDF is scanned (no readable text) ────────────────────

def is_scanned_pdf(pages: list[dict]) -> bool:
    """
    A quick check to see if a PDF is scanned (image-only) rather than text-based.

    If we extracted very little text from a multi-page document, it's likely
    a scanned PDF. We warn the user rather than silently failing.

    WHY THIS MATTERS:
        A scanned PDF gives us blank or near-blank page extractions.
        If we try to embed empty text and store it, the system will seem
        to work but will return useless results. Better to detect early
        and tell the user.

    NOTE for Phase 2:
        Full handling of scanned PDFs requires OCR (Optical Character
        Recognition) using a library like pytesseract. That's Phase 2.
        For now, we detect and warn.
    """
    if not pages:
        return True

    total_text = " ".join([p["text"] for p in pages])
    avg_chars_per_page = len(total_text) / len(pages)

    # If average is under 100 chars per page, it's almost certainly scanned
    return avg_chars_per_page < 100


# ── Main ingest function: combines all steps ──────────────────────────────────

def ingest_pdf(file_path: str) -> dict:
    """
    The main function you call from the outside.
    Takes a PDF file path, runs all three steps, returns everything.

    Returns:
        {
            "chunks": [...],        # list of chunk dicts, ready for embedding
            "page_count": 12,       # how many pages were extracted
            "chunk_count": 45,      # how many chunks were created
            "is_scanned": False,    # True if PDF appears to be image-only
            "filename": "policy.pdf"
        }

    USAGE:
        from rag.ingest import ingest_pdf
        result = ingest_pdf("data/uploaded/home_policy_2024.pdf")
        chunks = result["chunks"]
    """
    filename = os.path.basename(file_path)

    # Step 1: Extract text
    pages = extract_text_from_pdf(file_path)

    # Step 2: Check if it's a scanned PDF
    scanned = is_scanned_pdf(pages)

    # Step 3: Chunk the pages
    chunks = chunk_pages(pages)

    return {
        "chunks": chunks,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "is_scanned": scanned,
        "filename": filename
    }
