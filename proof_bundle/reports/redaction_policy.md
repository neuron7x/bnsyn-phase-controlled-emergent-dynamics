# Redaction Policy

- Do not store secrets, API keys, or token-like values in evidence logs.
- Keep only command strings and non-sensitive stdout/stderr required for reproducibility.
- Treat unknown high-entropy strings as sensitive and redact before publication.
