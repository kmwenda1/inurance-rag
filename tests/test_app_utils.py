import os

from rag.app_utils import build_safe_upload_path, sanitize_filename, validate_upload


class FakeUpload:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size


def test_sanitize_filename_removes_path_components():
    cleaned = sanitize_filename("../../my policy?.PDF")
    assert ".." not in cleaned
    assert "/" not in cleaned
    assert cleaned.lower().endswith(".pdf")


def test_build_safe_upload_path_stays_in_directory(tmp_path):
    file_path, display_name = build_safe_upload_path(str(tmp_path), "../../unsafe.pdf")

    assert display_name == "unsafe.pdf"
    assert os.path.commonpath([str(tmp_path), file_path]) == str(tmp_path)
    assert os.path.basename(file_path) != display_name


def test_validate_upload_rejects_non_pdf_and_oversize():
    ok, err = validate_upload(FakeUpload("notes.txt", 100), max_upload_size_bytes=1000)
    assert ok is False
    assert err is not None

    ok, err = validate_upload(FakeUpload("policy.pdf", 10_000), max_upload_size_bytes=1000)
    assert ok is False
    assert "maximum" in err.lower()

    ok, err = validate_upload(FakeUpload("policy.pdf", 800), max_upload_size_bytes=1000)
    assert ok is True
    assert err is None
