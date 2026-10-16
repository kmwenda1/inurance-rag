# Contributing

## Setup

```bash
cd /home/runner/work/inurance-rag/inurance-rag
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`GROQ_API_KEY` is only needed for answer generation tests/manual app usage.

## Development expectations

- Keep `streamlit run app.py` as the main runtime entrypoint.
- Add/adjust tests for behavior changes under `tests/`.
- Avoid exposing raw exceptions or secrets in UI messages.
- Keep upload/storage paths safe and deterministic.

## Checks before PR

```bash
ruff check .
pytest -q
```

## Pull request guidelines

- Keep changes focused and reviewable.
- Document user-visible behavior changes in `CHANGELOG.md`.
- If data flow/security assumptions changed, update `README.md` and `SECURITY.md`.
