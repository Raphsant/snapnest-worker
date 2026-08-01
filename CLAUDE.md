# CLAUDE.md — STC Zombie Hour pipeline worker

Python worker service for the STC (Stocks Trading Club) Zombie Hour video
pipeline: turns long trading-session recordings into finished 9:16 vertical
shorts. Polls SQS, processes PipelineJob rows through staged execution,
writes state to the shared Postgres DB (same RDS the snapnest-backend
NestJS app uses — the PipelineJob table is Prisma-managed; this repo NEVER
runs migrations, it only reads/writes rows).

## Stack
- Python managed with uv. ALWAYS `uv venv` / `uv pip` — the system Python
  is a uv-managed standalone build; `python3 -m venv` produces broken envs.
- psycopg for Postgres. Connection strings: psycopg requires
  `sslmode=require` and rejects `schema=public`. (Prisma-side strings use
  `sslmode=no-verify` — never copy one dialect into the other.)
- AWS: SQS queue `snapnest-pipeline-jobs` (3600s visibility timeout), S3
  bucket `snapnest-uploads-dev-rs`, AWS Transcribe (vocabulary
  `stc-trading-es`, profanity filter `stc-profanity-es`).
- Anthropic API for the curation and creative stages.
- Higgsfield CLI (`seedance_2_0`) for paid AI video generation.
  `generate create --json` returns a BARE ARRAY `["<id>"]`, not an object.
  Poll statuses must include `waiting` in the recognized set.
- ffmpeg (homebrew ffmpeg-full tap — the `subtitles`/libass filter is
  required). Normalization chains must set `setsar=1` explicitly.

## Pipeline stages (in order)
ingest → transcription → curation → build → [AWAITING_MANIFEST_APPROVAL
human gate] → cut → creative → [CREATIVE_APPROVED gate] → higgsfield
generate → assemble. A stage that sets a non-terminal "paused" status ends
the run for that job without marking COMPLETED.

## Iron rules
1. AI stages generate footage and prompts; the deterministic assembler
   applies ALL branding, text overlays, and captions. Never put logos or
   burned text in generation prompts.
2. Higgsfield generation costs real credits: per-asset checkpointing is
   mandatory — never re-generate an asset that already has a completed
   generation recorded. Nothing generates without the manifest + creative
   approval gates passed.
3. AI never writes timestamps or transcript text — scripts extract them
   from the SRT verbatim.
4. Main clip audio is never speed-adjusted or re-transcribed.
5. Fonts live in worker/fonts/ (Montserrat ExtraBold: captions/close_text;
   Anton: hook_text).

## Working with the operator
- One file at a time, show diffs, pause for approval. Never run against
  the DB, SQS, or Higgsfield without explicit go-ahead.
- Unit tests are not integration proof: the operator verifies with real
  trigger→queue→worker runs before anything is considered done.
- End every task with a test plan the operator runs manually.
