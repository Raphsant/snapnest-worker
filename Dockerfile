FROM python:3.11-slim

# ffmpeg from Debian apt. NOTE: the stock apt build may be compiled WITHOUT
# libass, so subtitle burn-in (the `subtitles`/`ass` filters) will not work.
# When we add subtitle stages, swap this for a libass-enabled ffmpeg — e.g. a
# static build (johnvansickle.com/ffmpeg) or a custom compile with --enable-libass.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching. pyproject drives the install.
COPY pyproject.toml ./
COPY worker ./worker
RUN pip install --no-cache-dir .

# Single-threaded long-poll loop; no ports to expose.
CMD ["python", "-m", "worker.main"]
