# Deploying the STC pipeline worker on TrueNAS / Dockge

The worker runs 24/7 as a single Docker Compose service. It builds from this
repo (nothing unrebuildable is baked into the image), reads all secrets from
`.env`, and authenticates to Higgsfield via a bind-mounted credentials dir.

**Container identity** (defined in the [Dockerfile](Dockerfile), referenced
throughout): runs as user `worker`, **UID/GID `10001`**, `HOME=/home/worker`.

## Stack layout

Check the repo out on the TrueNAS host and point a Dockge stack at it (or copy
it into the stack dir). The stack dir must end up with:

```
docker-compose.yml      # from the repo
.env                    # you create it (see step 2)
higgsfield-config/      # you create + seed it (see step 1)
```

`higgsfield-config/` and `.env` are git-ignored and docker-ignored — they never
enter a commit or an image layer.

## 1. Seed Higgsfield credentials (one-time)

The Higgsfield CLI reads/writes `config.json` + `credentials.json` under
`$HOME/.config/higgsfield`. Copy the two files from the Mac where you first
logged in — **skip `.lock`** — into the host `higgsfield-config/` dir, then fix
ownership so the container user can refresh the token in place.

On the Mac:

```bash
scp ~/.config/higgsfield/config.json ~/.config/higgsfield/credentials.json \
    truenas:/path/to/stack/higgsfield-config/
```

On the TrueNAS host, as root (in the stack dir):

```bash
chown -R 10001:10001 higgsfield-config
chmod 700 higgsfield-config
chmod 600 higgsfield-config/config.json higgsfield-config/credentials.json
```

> The mount is **read-write on purpose** — the CLI rewrites `credentials.json`
> on every token refresh. A read-only mount breaks after the first refresh.

## 2. Configure `.env`

Copy `.env.example` to `.env` and fill in the required values. Re-read the
`DATABASE_URL` warning in that file: psycopg needs `sslmode=require` and **no**
`schema` param — it is **not** the Prisma `no-verify` string.

## 3. Build & start via Dockge

In Dockge, deploy the stack (Dockge builds the image because compose declares a
`build:` context). Equivalent CLI, run in the stack dir:

```bash
docker compose up -d --build
```

The scratch volume (`snapnest-jobs`) is created automatically. If you switched
to the TrueNAS bind-mount alternative in `docker-compose.yml`, that dir must
also be writable by UID `10001`.

## 4. Smoke test — one throwaway YouTube job

Trigger a throwaway YouTube job from the dashboard, then watch the logs
(Dockge's log pane, or `docker logs -f snapnest-worker`). A healthy first run
shows, in order:

1. `Worker started; polling …` then `Processing job <id> from entry stage download`
2. `download[<id>]: fetching <url>` → `download[<id>]: fetched NNN MB; uploading to pipeline/<id>/source/source.mp4`
3. `download[<id>]: complete; sourceS3Key=pipeline/<id>/source/source.mp4`
4. ingest stage logs, and the job finally pausing at **`AWAITING_MANIFEST_APPROVAL`**.

Reaching the manifest gate proves the three things this box couldn't prove
before: **yt-dlp works from this IP**, the **RDS security group allows this
host** (DB reads/writes succeeded), and **S3 writes succeed**.

## The one-consumer rule

**Exactly one worker may consume `snapnest-pipeline-jobs`.** Two consumers means
double-processed jobs, races, and wasted Higgsfield credits. Once this worker is
verified healthy, **stop the Mac worker** (Ctrl-C / kill the local
`python -m worker.main`) and leave only the TrueNAS one running.

## Known future failure modes

**1. Generate stage fails with Higgsfield auth errors (typically after weeks
idle).** Normal token refresh is automatic (that's why the mount is RW); this
only happens when the *refresh token itself* has expired. Fix: re-authenticate
the Higgsfield CLI on the Mac, then re-copy `config.json` + `credentials.json`
into `higgsfield-config/` (redo the `chown`/`chmod` from step 1) and re-drive
the affected job.

**2. Worker can't reach Postgres (psycopg connection errors).** The RDS security
group allowlists this box's public IP; if the site's IP rotated, the connection
starts failing. Fix: update the RDS security-group inbound rule to the new
public IP.
