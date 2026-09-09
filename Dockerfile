# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM ghcr.io/astral-sh/uv:0.8.22-python3.12-bookworm-slim@sha256:28df4bbd896cf66a224f2e0cb22240a9a2b9803a3a13519bcadf2e9fdd68c632 AS builder

ENV UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group agent --group ui --no-install-project

COPY app.py ./
COPY src ./src
COPY data/fixtures ./data/fixtures
RUN uv sync --frozen --no-dev --group agent --group ui \
    && .venv/bin/python -c "from chest_xray_evidence_assistant.fixtures import load_fixture_manifest; load_fixture_manifest()"

FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv
COPY --chown=10001:10001 app.py ./
COPY --chown=10001:10001 src ./src
COPY --chown=10001:10001 data/fixtures ./data/fixtures
USER 10001:10001

EXPOSE 7860

HEALTHCHECK --interval=2s --timeout=2s --start-period=10s --retries=15 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=1).read(1)"]

CMD ["/app/.venv/bin/python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
