# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately by opening a private security advisory in GitHub (preferred) or contacting the repository owner directly. Do not publish exploit details in public issues.

Include:
- affected file/feature
- reproduction steps
- impact assessment
- suggested mitigation (if available)

## Security notes for this prototype

- `GROQ_API_KEY` must be stored in environment secrets, never committed.
- Uploaded documents are written to local storage.
- Retrieved document excerpts are sent to a third-party LLM provider (Groq) for answer generation.
- Do not use this prototype for regulated/sensitive data without additional controls (auth, encryption, retention policy, audit logging, and legal review).

## Scope limits

This is a prototype and does not currently implement enterprise-grade identity, tenancy isolation, or full data governance controls.
