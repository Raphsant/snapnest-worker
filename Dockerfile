# STC Zombie Hour pipeline worker — a 24/7 single-threaded SQS consumer.
# Build from the repo root (see DEPLOY.md):  docker compose build
#
# Design notes:
#  * Third-party deps install into /opt/venv from the FROZEN uv.lock (no
#    resolution drift). The worker package itself is NOT installed — it runs
#    from source at /app/worker, so prompts/ and fonts/ resolve relative to the
#    package (fonts are not shipped as wheel package-data).
#  * Runs as a non-root user (UID/GID 10001) with a writable HOME, because the
#    Higgsfield CLI refreshes credentials.json in place under
#    $HOME/.config/higgsfield (bind-mounted read-write at run time).

FROM python:3.11-slim

# Logs stream to the container's std streams unbuffered; don't attempt to write
# .pyc into the root-owned source tree at run time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# --- System packages ------------------------------------------------------
# ffmpeg MUST be a libass build (captions burn via the `subtitles` filter) with
# libfreetype (`drawtext` overlays); fontconfig lets libass load the bundled
# faces via the subtitles filter's fontsdir= option. curl + ca-certificates
# fetch the pinned Higgsfield CLI below. The build FAILS HERE if this ffmpeg is
# missing either filter, rather than failing at run time on a paid job.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ffmpeg \
        fontconfig \
        curl \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && { ffmpeg -hide_banner -filters | grep -qw subtitles \
        || { echo "FATAL: ffmpeg lacks the 'subtitles' filter (libass missing)"; exit 1; }; } \
 && { ffmpeg -hide_banner -filters | grep -qw drawtext \
        || { echo "FATAL: ffmpeg lacks the 'drawtext' filter (libfreetype missing)"; exit 1; }; }

# --- Higgsfield CLI (pinned Go binary) ------------------------------------
# Public GoReleaser build: github.com/higgsfield-ai/cli. The worker shells out
# to `higgsfield`; the archive ships a single binary named `hf` at its root
# (verified), so we symlink. Pinned to 1.1.13 — the version all live
# verification ran against; do NOT bump here. The archive is sha256-verified
# against the release checksums.txt; a mismatch fails the build.
#   linux_amd64 sha256: faf3792cf8edad262196560aad4ced8981df3b41df53593a782bf0c9d84ca001
#   linux_arm64 sha256: e607237418a14d1cd1585e9a59ddc13c99a01594ea152787a136dddf9194ac4d
#     (arm64 host: override both --build-arg HF_ARCH=arm64 and --build-arg HF_SHA256=...)
ARG HF_VERSION=1.1.13
ARG HF_ARCH=amd64
ARG HF_SHA256=faf3792cf8edad262196560aad4ced8981df3b41df53593a782bf0c9d84ca001
RUN curl -fsSL -o /tmp/hf.tar.gz \
      "https://github.com/higgsfield-ai/cli/releases/download/v${HF_VERSION}/hf_${HF_VERSION}_linux_${HF_ARCH}.tar.gz" \
 && echo "${HF_SHA256}  /tmp/hf.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/hf.tar.gz -C /usr/local/bin hf \
 && ln -s /usr/local/bin/hf /usr/local/bin/higgsfield \
 && rm /tmp/hf.tar.gz \
 && higgsfield --version

# --- Python dependencies via uv (reproducible, from uv.lock) --------------
# Pin uv to the version that produced uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app
# Deps-only layer: caches until pyproject.toml/uv.lock change. --no-install-project
# skips the worker package (it runs from source); --no-dev drops pytest/mypy/stubs.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# --- Non-root runtime user + writable dirs --------------------------------
# UID/GID 10001: the host Higgsfield-credentials dir and the scratch volume
# must be owned by this UID (see DEPLOY.md and docker-compose.yml). The scratch
# dir is chowned here so the named volume inherits worker ownership on first mount.
ENV HOME=/home/worker \
    JOB_WORKSPACE_ROOT=/var/lib/snapnest/jobs
RUN groupadd -g 10001 worker \
 && useradd -u 10001 -g 10001 -m -d /home/worker worker \
 && mkdir -p /home/worker/.config/higgsfield "${JOB_WORKSPACE_ROOT}" \
 && chown -R 10001:10001 /home/worker /var/lib/snapnest

# --- Application source ----------------------------------------------------
COPY worker ./worker
# Fonts are MANDATORY for the assemble stage (Anton -> hook text; Montserrat
# ExtraBold -> captions + close text). Assert the exact faces are present so the
# build FAILS LOUDLY instead of shipping a worker that silently falls back to a
# default face at assemble time.
RUN test -f worker/fonts/Anton-Regular.ttf && test -f worker/fonts/Montserrat-ExtraBold.ttf \
 || { echo "FATAL: worker/fonts/ must contain Anton-Regular.ttf and Montserrat-ExtraBold.ttf"; exit 1; }

USER worker

# Single-threaded SQS long-poll loop; no ports.
CMD ["python", "-m", "worker.main"]
