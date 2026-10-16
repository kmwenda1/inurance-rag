import os
import re
import uuid
from pathlib import Path

MAX_UPLOAD_SIZE_MB_DEFAULT = 20


def get_max_upload_size_bytes() -> int:
    raw = os.getenv("MAX_UPLOAD_SIZE_MB", str(MAX_UPLOAD_SIZE_MB_DEFAULT)).strip()
    try:
        size_mb = int(raw)
    except ValueError:
        size_mb = MAX_UPLOAD_SIZE_MB_DEFAULT
    size_mb = max(size_mb, 1)
    return size_mb * 1024 * 1024


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe filename while preserving extension when possible."""
    base = Path(filename or "upload.pdf").name
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(base).stem).strip("._")
    ext = re.sub(r"[^A-Za-z0-9.]", "", Path(base).suffix.lower())
    if not ext:
        ext = ".pdf"
    if not stem:
        stem = "upload"
    return f"{stem}{ext}"


def build_safe_upload_path(upload_dir: str, original_filename: str) -> tuple[str, str]:
    """Return absolute safe storage path and original display name."""
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = sanitize_filename(original_filename)
    unique_name = f"{Path(safe_name).stem}_{uuid.uuid4().hex[:12]}{Path(safe_name).suffix}"
    return os.path.abspath(os.path.join(upload_dir, unique_name)), Path(original_filename).name


def validate_upload(uploaded_file, max_upload_size_bytes: int) -> tuple[bool, str | None]:
    """Validate type and size for uploaded Streamlit file objects."""
    name = (uploaded_file.name or "").lower()
    if not name.endswith(".pdf"):
        return False, "Only PDF files are supported."

    file_size = getattr(uploaded_file, "size", None)
    if file_size is None:
        try:
            file_size = len(uploaded_file.getbuffer())
        except Exception:
            file_size = 0

    if file_size <= 0:
        return False, "Uploaded file is empty."

    if file_size > max_upload_size_bytes:
        max_mb = max_upload_size_bytes // (1024 * 1024)
        return False, f"File exceeds the maximum allowed size ({max_mb} MB)."

    return True, None
