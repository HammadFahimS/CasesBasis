# CasesBasis — Production Deployment

**Status: LIVE at https://casesbasis.com** (first deployed 2026-07-13).

This file is the authoritative record of *what is actually running in production* and
how to operate/change it. It intentionally reflects the real, as-built setup — which
**diverged from `deploy/README.md`** (that older runbook assumed a Cloudflare Tunnel; we
shipped a Caddy + Let's Encrypt front instead — see "Why this shape" below). If you change
the deployment, update this file.

---

## 1. Architecture

```
  Browser ──HTTPS(443)──> Caddy ──HTTP──> app:8000 (FastAPI/uvicorn, 1 worker)
  (casesbasis.com)     (LE cert)   (internal        │
                       auto-renew)  docker net)      └──> falkordb:6379 (internal)
        http(80) ─301─> https
```

- **Caddy** (`caddy:2`) is the only thing bound to the host's public ports (80, 443).
  It terminates TLS with a real Let's Encrypt certificate and reverse-proxies to the app.
- **app** (FastAPI) has **no published host port** — reachable only on the internal Docker
  network, only through Caddy.
- **falkordb** (knowledge graph) — internal only, never exposed.
- Everything runs via Docker Compose on a single DigitalOcean droplet.

### Why this shape (not the Cloudflare Tunnel in `deploy/README.md`)
The Cloudflare API token we have can edit **DNS only** — it **cannot** manage Cloudflare
Tunnels or change the zone's SSL/TLS mode. A proxied (orange-cloud) record without being
able to set SSL=Full(strict) would break TLS. So instead of the tunnel, we put **Caddy** on
the droplet to get browser-trusted HTTPS fully automatically. This matters because the
**voice feature needs the browser mic, and `getUserMedia` only works over real HTTPS** — a
self-signed cert or plain HTTP would break it. Caddy's Let's Encrypt cert makes the live
site behave exactly like localhost for the mic.

---

## 2. Server

- **Provider:** DigitalOcean droplet
- **Public IP:** `192.81.214.4`
- **Specs:** Ubuntu 24.04, **4 GB RAM / 2 vCPU**, ~77 GB disk, + **2 GB swap** (added as an OOM safety net)
- **SSH:** `ssh root@192.81.214.4` using the local key `~/.ssh/id_ed25519`
  (passphrase-protected; must be loaded in the agent: `ssh-add --apple-use-keychain ~/.ssh/id_ed25519`)
- **App location on box:** `/root/casesbasis`
- **Docker:** installed via get.docker.com, enabled on boot (`systemctl enable docker`) → the
  stack comes back automatically after a reboot.

**Sizing note:** the original droplet was 512 MB and would OOM instantly (CPU torch + an
835 MB FAISS index loaded into RAM + FalkorDB). 4 GB is the practical floor. Idle usage is
~2.2 GB (app ≈ 2.0 GiB incl. the index, falkordb ≈ 100 MiB, caddy ≈ 55 MiB), leaving ~1.4 GB
free + swap. Do **not** downsize below 4 GB.

---

## 3. How the code got there (NOT git-cloned)

The GitHub repo (`FazalMahmoodKhan/CasesBasis`) is **private**, and we deliberately did not
put GitHub credentials on the droplet. The code was **rsync'd from the local `main` checkout**
over SSH. The container image is then **built on the droplet** from that code.

Deploy scaffolding lives on `main` (commit `9aa16e1`): `Dockerfile`, `requirements.txt`
(163-pkg lock from the working venv), `docker-compose.prod.yml`, `.dockerignore`,
`.env.production.example`, `deploy/README.md`.

---

## 4. The running stack (Docker Compose)

Compose files on the droplet in `/root/casesbasis`:
- `docker-compose.prod.yml` — committed base: `app`, `falkordb`, and a `cloudflared` service
  (the tunnel — **NOT used in prod**).
- `docker-compose.override.yml` — **droplet-only, not in git**. Adds the `caddy` service
  (ports 80/443, mounts `./Caddyfile`, `caddy_data`/`caddy_config` volumes).
- `Caddyfile` — **droplet-only, not in git**:
  ```
  casesbasis.com {
      reverse_proxy app:8000
  }
  ```

**The stack is brought up naming only the services we want (this excludes `cloudflared`):**
```bash
cd /root/casesbasis
docker compose -f docker-compose.prod.yml -f docker-compose.override.yml up -d app falkordb caddy
```

Compose merges base + override; `depends_on` + healthchecks enforce ordering
(falkordb healthy → app healthy → caddy). All three services are `restart: unless-stopped`
with json-file log rotation.

> App healthcheck has a **90 s `start_period`** (migrations + seeding + warming the 835 MB
> CXR FAISS index). `/health` usually returns 200 within a few seconds; the container may show
> "health: starting" for up to 90 s regardless — that's normal.

---

## 5. TLS / HTTPS

- Caddy auto-obtained a **Let's Encrypt** cert for `casesbasis.com` (ACME `tls-alpn-01`).
- Cert + account data persist in the `caddy_data` Docker volume; **Caddy auto-renews** — no cron needed.
- Port 80 exists only for ACME + the automatic `http → https` 301/308 redirect. The app is
  **never** served over plain HTTP.

---

## 6. DNS

- **A-record:** `casesbasis.com → 192.81.214.4`, **grey-cloud (DNS-only / not proxied)**,
  created via the Cloudflare API.
- **Keep it grey.** Flipping to proxied (orange) would put Cloudflare in the TLS path, and the
  token can't set the SSL mode — it would break HTTPS/cert issuance. Going proxied later is a
  deliberate step that needs an SSL-mode change (dashboard or an upgraded token) first.
- **No `www` record** yet, and **no email/MX** (see §10 follow-ups).

### Cloudflare token / IDs (for future API calls)
- Zone `casesbasis.com`: `51adf893a05adb718f2e715492b4d884`
- Account "Fazal Mahmood Khan": `7ffbef4f907f1afc51f635b6f8496717`
- Token scope (as used): `#dns_records:edit`, `#dns_records:read`, `#zone:read`.
  **Cannot** do: Tunnels (`cfd_tunnel`), SSL settings, Email Routing, account-level ops.
- ⚠️ The token string was pasted into a chat log during setup — **it should be rotated.**
  (Do not store token secrets in this file or the repo.)

---

## 7. Secrets — `.env.production`

- Lives **only** on the droplet at `/root/casesbasis/.env.production` (perms `600`).
- Built from the local `.env` (all provider keys) + `ENV=production` + a freshly generated
  `APP_SECRET`. Inline `#` comments were stripped (Compose's env_file parser keeps them literally).
- **Never committed, never baked into the image, never printed.** `.env` and `.env.production`
  are git-ignored; only `.env.production.example` (a value-less template) is tracked.
- Compose's base file also sets these as container env (overriding the file):
  `ENV=production`, `LLM_PROVIDER=cerebras`, `DB_PATH=/app/var/casesbasis.db`,
  `FALKORDB_HOST=falkordb`, `FALKORDB_PORT=6379`, `ALLOWED_ORIGINS=https://casesbasis.com`.
- To edit a key: `nano /root/casesbasis/.env.production` then
  `docker compose -f docker-compose.prod.yml -f docker-compose.override.yml up -d app`.

---

## 8. Data (FAISS indexes + images)

Runtime data is git-ignored and was rsync'd separately into `/root/casesbasis/data`,
bind-mounted read-only into the app (`./data:/app/data:ro`):
- `data/cxr` (~9.5 MB, raw CXR images used by the legacy CXR payloads)
- `data/cxr_index` (~835 MB FAISS index + `metadata.parquet`; **loaded into RAM at startup**)
- `data/ptbxl_index` (~26 MB; loaded lazily)
- **Skipped:** `data/ptb-xl` (3.2 GB raw) — PTB-XL is disabled in code, not needed at runtime.

Embedding models (`all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`) are **baked into the image** at
build time, so a live boot never depends on HuggingFace.

---

## 9. Database (SQLite)

- Prod DB is SQLite at `/app/var/casesbasis.db` inside the container, on the **`app_state`
  named volume** (host path: `/var/lib/docker/volumes/casesbasis_app_state/_data/`), owned by
  uid `10001` (the container's `appuser`).
- `run_migrations()` runs on every startup (idempotent, tracked in `schema_migrations`).
  Gold-case + qbank seeding is insert-only.
- **The prod DB was replaced with the local DB on 2026-07-13** (user's data: 30 qbank cases,
  2 simulated cases, 3 resident accounts, plus session history). The pre-existing fresh prod DB
  was backed up on the droplet: `/root/prod-db-backup-20260713-052046.db`.

### Push a fresh local DB to prod (repeatable procedure)
```bash
# 1. On the Mac: consistent snapshot (merges WAL; safe while the DB is open)
sqlite3 casesbasis.db "VACUUM INTO '/tmp/casesbasis_prod.db'"
sqlite3 /tmp/casesbasis_prod.db "PRAGMA integrity_check;"          # expect: ok
scp /tmp/casesbasis_prod.db root@192.81.214.4:/root/casesbasis/casesbasis_new.db

# 2. On the droplet: stop app, back up, swap, fix ownership, start
ssh root@192.81.214.4
cd /root/casesbasis
VOL=$(docker volume inspect casesbasis_app_state -f '{{.Mountpoint}}')
docker compose -f docker-compose.prod.yml -f docker-compose.override.yml stop app
cp "$VOL"/casesbasis.db /root/prod-db-backup-$(date +%Y%m%d-%H%M%S).db
cp /root/casesbasis/casesbasis_new.db "$VOL"/casesbasis.db
rm -f "$VOL"/casesbasis.db-wal "$VOL"/casesbasis.db-shm
chown 10001:10001 "$VOL"/casesbasis.db
docker compose -f docker-compose.prod.yml -f docker-compose.override.yml up -d app
```

### Back up the live prod DB
```bash
ssh root@192.81.214.4 'docker exec casesbasis-app sqlite3 /app/var/casesbasis.db ".backup /app/var/backup.db"'
ssh root@192.81.214.4 'docker cp casesbasis-app:/app/var/backup.db /root/prod-db-backup-$(date +%F).db'
```

---

## 10. Redeploy after code changes

The image is built on the droplet from rsync'd code. After changing code locally on `main`:

```bash
# From the Mac (repo root). NOTE: macOS ships old rsync 2.6.9 — do NOT use --info=progress2.
# The --exclude of docker-compose.override.yml + Caddyfile is CRITICAL: those are
# droplet-only (not in git), so without excluding them --delete would wipe them.
rsync -az --delete \
  --exclude='.git' --exclude='.venv' --exclude='data' --exclude='__pycache__' \
  --exclude='*.pyc' --exclude='.pytest_cache' --exclude='.env' --exclude='.env.production' \
  --exclude='*.db' --exclude='*.db-wal' --exclude='*.db-shm' --exclude='.DS_Store' \
  --exclude='docker-compose.override.yml' --exclude='Caddyfile' \
  ./ root@192.81.214.4:/root/casesbasis/

# On the droplet: rebuild + restart just the app (falkordb/caddy keep running)
ssh root@192.81.214.4 'cd /root/casesbasis && \
  docker compose -f docker-compose.prod.yml -f docker-compose.override.yml up -d --build app'
```

> ⚠️ rsync excludes `data`, `.env*`, and the DB — so a redeploy never clobbers prod data,
> secrets, or the FAISS indexes. If you add new data files, rsync them explicitly (see §8).

If the frontend/static changed, the same rebuild picks it up (`COPY . .` in the Dockerfile).

---

## 11. Day-2 operations

```bash
cd /root/casesbasis
C="docker compose -f docker-compose.prod.yml -f docker-compose.override.yml"

$C ps                       # status
$C logs -f app              # tail app logs
$C logs -f caddy            # tail caddy (TLS/cert) logs
$C restart app              # restart app only
$C up -d app                # recreate app (after env/code change)
$C down                     # stop everything (volumes persist)
$C -p casesbasis ... 	    # (compose project name is "casesbasis")

docker stats --no-stream    # live memory/CPU per container
curl -s https://casesbasis.com/health   # -> {"status":"ok"}
```

Health/verify from anywhere:
```bash
curl -sS -o /dev/null -w "%{http_code} tls=%{ssl_verify_result}\n" https://casesbasis.com/
```

---

## 12. Robustness (what keeps it up)

- **Auto-restart:** all containers `restart: unless-stopped` + Docker enabled on boot →
  survives crashes and full reboots.
- **Health-gated startup ordering** (falkordb → app → caddy).
- **Log rotation:** json-file, 10 MB × (5 app/db, 3 caddy) — disk can't fill from logs.
- **Persistent volumes:** `app_state` (SQLite), `falkordb_data`, `caddy_data`/`caddy_config`
  (certs) — survive redeploys/`down`.
- **Swap** (2 GB) as an OOM cushion.
- **Single worker on purpose:** session state (`SESSIONS`, `_SESSIONS`) is in-process — do
  **not** raise uvicorn workers or scale the app horizontally; scale vertically (bigger droplet).
- **Cert auto-renew** by Caddy.

---

## 13. Gotchas / troubleshooting

- **502 from the site** → app is down/unhealthy. `docker compose ... logs -f app`. It may be
  mid-startup (90 s index warmup) or OOM (check `docker stats` / `dmesg | grep -i oom`).
- **Cert didn't issue / renew** → check `docker logs casesbasis-caddy`; ensure ports 80/443 are
  open (no DO cloud firewall) and the A-record still points to the droplet and is **grey-cloud**.
- **Mic not working on the live site** → must be `https://` with a valid cert (it is). If you
  ever move behind Cloudflare-proxy with a broken SSL mode, the mic + `wss://` will fail.
- **rsync fails with a usage dump** → you used a flag the old macOS rsync (2.6.9) rejects (e.g.
  `--info=progress2`). Use plain `-az`.
- **Private repo can't clone on the droplet** → by design; deploy via rsync (§10).
- **DB owned by root after a manual copy** → the app (uid 10001) can't write. `chown 10001:10001`.

---

## 14. Follow-ups / TODO (user's explicit "fix later")

- [ ] **Rotate the Cloudflare API token** (it was pasted in a chat log).
- [ ] Proper **dev ↔ prod** separation (repos, keys, DBs).
- [ ] Optional: add **`www.casesbasis.com`** (A-record + it's already covered if added to the Caddyfile).
- [ ] Optional: go **Cloudflare-proxied** to hide the origin IP + add CDN/DDoS — requires setting
      SSL/TLS mode to **Full (strict)** first (needs dashboard or an upgraded token), otherwise TLS breaks.
- [ ] **Email**: no mailbox/MX set up for `hammad@casesbasis.com`. Options discussed: Zoho Mail
      (free real mailbox), Google Workspace (paid), or Cloudflare Email Routing (free forwarding).
      Whichever is chosen, the account is created by the user; DNS (MX/SPF/DKIM/DMARC) is added via the token.
- [ ] Consider moving off `root` SSH + off rsync-deploy to a proper CI/CD later.

---

## 15. Key facts index (quick reference)

| Thing | Value |
|---|---|
| URL | https://casesbasis.com |
| Droplet IP | 192.81.214.4 (Ubuntu 24.04, 4 GB/2 vCPU) |
| App dir | `/root/casesbasis` |
| Compose | `-f docker-compose.prod.yml -f docker-compose.override.yml` (services: `app falkordb caddy`) |
| DB (prod) | volume `app_state` → `/app/var/casesbasis.db` (`DB_PATH`) |
| Secrets | `/root/casesbasis/.env.production` (600, not in git) |
| TLS | Caddy + Let's Encrypt, auto-renew |
| DNS | grey-cloud A-record via CF token (zone `51adf893a05adb718f2e715492b4d884`) |
| Deploy method | rsync from local `main` + build on droplet |
| Deploy commit | `944674e` on `main` (2026-09-05 — foreign-body airway gold case + generic engine/nurse/consult mechanisms) |

Related: repo `deploy/README.md` (older tunnel-oriented runbook), `Dockerfile`,
`docker-compose.prod.yml`, and the auto-memory `deployment-status.md`.
