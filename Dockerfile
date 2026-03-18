FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN bash scripts/bootstrap.sh --venv .venv --ready-file .venv/.ready-dev --extras dev,test

RUN ./.venv/bin/python -m pytest -q tests/test_build_agent_feedback.py tests/test_check_canonical_phase_gates.py tests/test_compare_canonical_runs.py tests/test_synthesize_canonical_remediation.py

ENTRYPOINT ["make"]
CMD ["quickstart-smoke"]
