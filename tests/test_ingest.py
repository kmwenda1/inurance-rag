from rag import ingest


def test_chunk_pages_creates_overlapping_chunks():
    pages = [{"page_num": 1, "text": "abcdefghijklmnopqrstuvwxyz", "source": "doc.pdf"}]

    chunks = ingest.chunk_pages(pages, chunk_size=10, overlap=3)

    assert len(chunks) >= 3
    assert chunks[0]["text"] == "abcdefghij"
    assert chunks[1]["text"].startswith("hij")
    assert chunks[0]["chunk_id"] == "doc.pdf_page1_chunk0"


def test_is_scanned_pdf_detection():
    assert ingest.is_scanned_pdf([]) is True

    low_text_pages = [{"text": "x" * 20}, {"text": "x" * 30}]
    rich_text_pages = [{"text": "x" * 500}, {"text": "x" * 700}]

    assert ingest.is_scanned_pdf(low_text_pages) is True
    assert ingest.is_scanned_pdf(rich_text_pages) is False


def test_extract_text_from_pdf_uses_source_name(monkeypatch):
    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakePdf:
        def __enter__(self):
            self.pages = [FakePage("Page one text"), FakePage(""), FakePage("Page three")]
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(ingest.pdfplumber, "open", lambda _: FakePdf())

    pages = ingest.extract_text_from_pdf("/tmp/unsafe.pdf", source_name="original.pdf")
    assert len(pages) == 2
    assert pages[0]["source"] == "original.pdf"
    assert pages[1]["page_num"] == 3


def test_ingest_pdf_handles_malformed_pdf(monkeypatch):
    monkeypatch.setattr(ingest, "extract_text_from_pdf", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad pdf")))

    result = ingest.ingest_pdf("/tmp/corrupted.pdf")

    assert result["chunk_count"] == 0
    assert result["is_scanned"] is True
    assert result["error"] is not None
