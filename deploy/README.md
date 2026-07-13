# Deploying CasesBasis to casesbasis.com

Production architecture:

```
  Browser ──HTTPS──> Cloudflare edge ──Tunnel──> cloudflared ──> app:8000 (FastAPI)
  (casesbasis.com)     (TLS here)     (outbound)      │
                                                      └──> falkordb:6379 (internal)
```

The VPS has **no open inbound ports**. Cloudflare Tunnel dials *out* from the
box, so the app is never directly exposed and TLS is handled by Cloudflare.
API keys live only in `.env.production` on the server.

---

## What you need to provide

1. **A VPS** — Ubuntu 22.04/24.04, **≥ 4 GB RAM** (8 GB comfortable), ≥ 20 GB disk,
   Docker installed. Hetzner CPX21 / DigitalOcean 4 GB are fine.
2. **A Cloudflare API token** scoped with `Zone.DNS:Edit` +
   `Account.Cloudflare Tunnel:Edit` (the current token is read-only — see below).
   *You don't strictly need the token if you create the tunnel in the dashboard.*
3. **The API keys** for `.env.production` (Cerebras, ElevenLabs, Deepgram…).

---

## Step 1 — Create the Cloudflare Tunnel (one-time)

Zero Trust dashboard → **Networks → Tunnels → Create a tunnel** →
*Cloudflared* → name it `casesbasis`.

- Copy the **tunnel token** it shows you → this becomes `TUNNEL_TOKEN`.
- Under **Public Hostname**, add:
  - **Subdomain**: (blank) **Domain**: `casesbasis.com` → Service `HTTP` → `app:8000`
  - Add a second for `www` if you want.

That auto-creates the DNS records — no manual DNS editing needed.

## Step 2 — Provision the box

```bash
# On the VPS, as root/sudo:
curl -fsSL https://get.docker.com | sh
```

## Step 3 — Get the code + data onto the server

The image is built from source on the box. The ~860 MB of FAISS indexes / CXR
images (`data/`) are git-ignored, so copy them separately.

```bash
# On the VPS:
git clone <your repo url> casesbasis && cd casesbasis

# From your Mac — ship the runtime datasets (indexes + raw CXR images).
# PTB-XL raw (3.2 GB) is NOT needed — it's disabled in code.
rsync -av --progress \
  data/cxr data/cxr_index data/ptbxl_index \
  user@SERVER_IP:~/casesbasis/data/
```

## Step 4 — Configure secrets

```bash
# On the VPS, in ~/casesbasis:
cp .env.production.example .env.production
python3 -c "import secrets; print('APP_SECRET='+secrets.token_hex(32))"   # paste into the file
nano .env.production   # fill CEREBRAS_API_KEY, ELEVENLABS_*, DEEPGRAM_API_KEY, TUNNEL_TOKEN
```

`.env.production` is git-ignored and never leaves the server.

## Step 5 — Launch

```bash
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f app     # watch it boot + seed
```

First boot runs SQL migrations, seeds the gold case + qbank, and warms the CXR
FAISS index (embedding models are already baked into the image). Give it ~1–2 min;
`/health` goes green when ready. Then open **https://casesbasis.com**.

---

## Robustness (built in)

- **Auto-restart**: every container is `restart: unless-stopped` — a crash or a
  server reboot brings the whole stack back automatically.
- **Health-gated**: the tunnel only starts once the app reports healthy; the app
  only starts once FalkorDB is healthy.
- **Log rotation**: 10 MB × 5 files per container — logs can't fill the disk over
  months of uptime.
- **Persistent state**: SQLite DB and the FalkorDB graph live on named volumes;
  redeploys (`up -d --build`) keep the data.
- **Single worker on purpose**: session state is in-process; scaling workers would
  split sessions. Vertical (bigger box) is the scaling axis, not more workers.

## Day-2 operations

```bash
# Update after pushing code
git pull && docker compose -f docker-compose.prod.yml up -d --build

# Tail / restart / stop
docker compose -f docker-compose.prod.yml logs -f app
docker compose -f docker-compose.prod.yml restart app
docker compose -f docker-compose.prod.yml down          # stop (keeps volumes)

# Back up the SQLite DB
docker compose -f docker-compose.prod.yml exec app \
  sh -c 'sqlite3 /app/var/casesbasis.db ".backup /app/var/backup.db"'
docker cp casesbasis-app:/app/var/backup.db ./backup-$(date +%F).db
```

## Rotate the pasted Cloudflare token

The read-only token shared during setup was exposed in a chat log. After the
tunnel is up (which uses a separate *tunnel token*, not the API token), delete
that API token in Cloudflare → My Profile → API Tokens.
