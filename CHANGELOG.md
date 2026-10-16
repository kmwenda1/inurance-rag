# Changelog

## Unreleased

### Improved prototype readiness
- Kept Streamlit entrypoint as `streamlit run app.py` and improved startup reliability.
- Added lazy initialization for FastEmbed and Groq client so app import/startup does not require API key.
- Added safe upload validation (file type, configurable max size) and safe stored filenames.
- Added robust handling for malformed PDFs and provider failures with user-safe error messages.
- Aligned retriever confidence thresholds and documentation.
- Removed unsafe dynamic HTML rendering for generated answer/source text.
- Added pytest test suite for ingestion, retrieval confidence, generator failure paths, startup behavior, and upload/path safety.
- Added CI workflow and lightweight lint configuration.
- Updated repository documentation for setup, architecture, security, and contribution guidance.
