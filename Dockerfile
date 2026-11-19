# Step 15: one image that can run every command (API, stream, bot, training...).
#   docker build -t dalal-street-lab .
#   docker run --env-file .env -p 8000:8000 dalal-street-lab              # the API
#   docker run --env-file .env dalal-street-lab dalal-stream             # any other command

# ---------- Build stage: install everything into /venv ----------
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/uv
WORKDIR /src
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy VIRTUAL_ENV=/venv

COPY pyproject.toml uv.lock README.md ./
# The lockfile's Linux torch bundles CUDA (gigabytes of GPU code). Servers here have no GPU,
# so install the CPU-only build of the SAME torch version, and everything else from the lock.
RUN uv venv /venv \
 && uv export --frozen --no-dev --no-hashes --no-emit-project \
      | grep -viE '^(torch|triton|nvidia-|cuda-)' > /tmp/requirements.txt \
 && TORCH=$(grep -A1 '^name = "torch"$' uv.lock | sed -n 's/^version = "\(.*\)"/\1/p') \
 && uv pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH}" \
 && uv pip install -r /tmp/requirements.txt

COPY src ./src
RUN uv pip install --no-deps .

# ---------- Runtime stage: just Python, the venv and the files commands need ----------
FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 dalal
COPY --from=build /venv /venv
WORKDIR /app
COPY alembic.ini watchlist.toml ./
COPY migrations ./migrations
ENV PATH=/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LOG_FORMAT=json \
    HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache && chown -R dalal /app
USER dalal
EXPOSE 8000 9101
CMD ["dalal-api"]
