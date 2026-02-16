# PersonaPlex -- GCP Cloud Run Deployment Guide

## What the Code Handles Automatically vs What You Do Manually

### The code/script handles ALL of this automatically:

| What | How |
|------|-----|
| Model download (~14GB weights) | `hf_hub_download()` at container startup, accelerated by `hf_transfer` |
| Voice prompts download | `voices.tgz` from HuggingFace, extracted automatically |
| React frontend download | `dist.tgz` from HuggingFace, extracted automatically |
| Text tokenizer download | `tokenizer_spm_32k_3.model` from HuggingFace |
| Model loading to GPU | `model_manager.py` loads Mimi + Moshi LM to CUDA at startup |
| CUDA graph warmup | 4 dummy iterations to compile CUDA graphs |
| Health check endpoint | `/health` returns 503 while loading, 200 when ready |
| Twilio inbound webhook | `/twilio/inbound` returns TwiML automatically |
| Audio format conversion | mulaw 8kHz <-> PCM 24kHz handled by `audio_converter.py` |
| Conversation recording | Auto-saved as WAV in `/app/recordings/` |
| GCP APIs enablement | `setup-gcp.sh` enables all required APIs |
| Artifact Registry repo | Created by `setup-gcp.sh` |
| Secret Manager setup | Twilio creds stored as secrets by `setup-gcp.sh` |
| IAM permissions | Service account access to secrets granted by `setup-gcp.sh` |
| Docker build + push | Done via Google Cloud Build by `setup-gcp.sh` (no local Docker needed) |
| Cloud Run deployment | Full `gcloud run deploy` with GPU, memory, concurrency |
| SERVER_URL configuration | Auto-detected and set after deployment |

### You MUST do these manually (one-time):

| Step | What | Details |
|------|------|---------|
| 1 | **Install gcloud CLI** | https://cloud.google.com/sdk/docs/install |
| 2 | **Authenticate gcloud** | `gcloud auth login` and `gcloud config set project pgai-personaplex-speech` |
| 3 | **HF token in Secret Manager** | Verify: `gcloud secrets describe hf-token --project=pgai-personaplex-speech` |
| 4 | **Have Twilio credentials ready** | Account SID, Auth Token, and a phone number |
| 5 | **Run the setup script** | `./deploy/setup-gcp.sh` (it will prompt for Twilio creds) |
| 6 | **Configure Twilio webhook** | In Twilio Console, point your number's voice webhook to `{SERVICE_URL}/twilio/inbound` |

That's it. Steps 1-3 are one-time setup. Step 4-5 is running the script. Step 6 is a Twilio console click.

---

## Quick Start

```bash
# 1. Make sure gcloud and docker are installed and authenticated
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Verify your HF token is in Secret Manager (you said it's already there)
gcloud secrets describe hf-token

# 3. Have your Twilio creds ready, then run:
chmod +x deploy/setup-gcp.sh
./deploy/setup-gcp.sh
```

The script will:
- Prompt for Twilio credentials if not set as env vars
- Enable APIs, create repos, store secrets
- Build the Docker image and push it
- Deploy to Cloud Run with an NVIDIA L4 GPU
- Wait for the service to become healthy
- Print the service URL and next steps

---

## How Model Download Works on Cloud Run

When the container starts on Cloud Run, the following happens automatically:

```
Container starts
  |
  +--> FastAPI lifespan begins
  |      |
  |      +--> model_manager.load() called
  |      |      |
  |      |      +--> hf_hub_download("config.json")           [~1s, cached after first run]
  |      |      +--> hf_hub_download("tokenizer-*.safetensors") [~2s, Mimi codec, ~200MB]
  |      |      +--> loaders.get_mimi() x2                     [~5s, build model on GPU]
  |      |      +--> hf_hub_download("tokenizer_spm_32k_3.model") [<1s]
  |      |      +--> hf_hub_download("model.safetensors")      [~30-60s, 14GB, hf_transfer accelerated]
  |      |      +--> loaders.get_moshi_lm()                    [~10-20s, load to GPU]
  |      |      +--> hf_hub_download("voices.tgz")             [~2s]
  |      |      +--> hf_hub_download("dist.tgz")               [~1s, React frontend]
  |      |
  |      +--> model_manager.warmup()                           [~10s, CUDA graph compilation]
  |      |
  |      +--> /health returns 200 (ready)
  |
  +--> Server accepts requests on :8080
```

**Total cold start: ~90-120 seconds** (dominated by the 14GB model download).

With `min-instances: 0` (the default), instances scale to zero when idle to save cost. First request after scale-down incurs the full cold start. Set `MIN_INSTANCES=1` before running the script to keep an instance warm.

**HF cache**: Downloads are cached at `/root/.cache/huggingface`. On Cloud Run, this cache is ephemeral (lost when instance is replaced). For persistent caching, you could mount a GCS FUSE volume, but the download with `hf_transfer` is fast enough that it's not strictly necessary.

---

## Port Configuration

Everything runs on **port 8080** (Cloud Run's default):

| Component | Port | Notes |
|-----------|------|-------|
| FastAPI (uvicorn) | 8080 | Main server |
| Dockerfile EXPOSE | 8080 | |
| docker-compose | 8080:8080 | |
| Cloud Run containerPort | 8080 | |
| Health checks | 8080 | `/health` endpoint |
| Legacy server.py | 8998 | Only for local standalone use, NOT used by Docker |

---

## Endpoints Reference

Once deployed, your service has these endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check. Returns `{"status": "healthy", ...}` |
| `/docs` | GET | Auto-generated API documentation (Swagger UI) |
| `/` | GET | React frontend (if available) |
| `/api/chat` | WebSocket | React frontend binary protocol (Opus audio) |
| `/twilio/inbound` | POST | Twilio voice webhook -- returns TwiML |
| `/twilio/outbound` | POST | Initiate outbound call. Query params: `to`, `voice_prompt`, `text_prompt` |
| `/twilio/media-stream` | WebSocket | Twilio Media Streams bidirectional audio |

---

## Configuration via Environment Variables

All configuration is via environment variables. The setup script handles these, but for reference:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | -- | HuggingFace API token (via Secret Manager) |
| `DEVICE` | No | `cuda` | `cuda` or `cpu` |
| `HF_REPO` | No | `nvidia/personaplex-7b-v1` | HuggingFace model repo |
| `HF_HUB_ENABLE_HF_TRANSFER` | No | `1` | Enable fast Rust-based downloads |
| `NO_TORCH_COMPILE` | No | `1` | Disable torch.compile |
| `CPU_OFFLOAD` | No | `0` | Set to `1` for CPU offloading |
| `SERVER_URL` | For Twilio | -- | Public URL (auto-set by setup script) |
| `TWILIO_ACCOUNT_SID` | For Twilio | -- | Via Secret Manager |
| `TWILIO_AUTH_TOKEN` | For Twilio | -- | Via Secret Manager |
| `TWILIO_PHONE_NUMBER` | For Twilio | -- | Via Secret Manager |
| `DEFAULT_VOICE_PROMPT` | No | `NATF2.pt` | Voice for Twilio calls |
| `DEFAULT_TEXT_PROMPT` | No | `` | System prompt for Twilio calls |
| `RECORDINGS_DIR` | No | `/app/recordings` | Where to save call recordings |

---

## Useful Commands After Deployment

```bash
# View live logs
gcloud run logs read personaplex --region=us-east4 --limit=50

# Stream logs
gcloud beta run logs tail personaplex --region=us-east4

# Check health
curl https://YOUR_SERVICE_URL/health

# Make a test outbound call
curl -X POST 'https://YOUR_SERVICE_URL/twilio/outbound?to=+1YOURNUMBER&voice_prompt=NATF2.pt&text_prompt=Hello'

# Redeploy after code changes
gcloud builds submit . --tag=IMAGE_TAG:latest --project=YOUR_PROJECT
gcloud run deploy personaplex --image=IMAGE_TAG:latest --region=us-east4

# Scale to zero when not in use (save costs)
gcloud run services update personaplex --region=us-east4 --min-instances=0

# Delete everything
gcloud run services delete personaplex --region=us-east4
```

---

## GPU Notes

- Cloud Run supports **NVIDIA L4 only** (24GB VRAM)
- PersonaPlex needs ~19GB VRAM -- L4's 24GB is sufficient
- For A100 (40/80GB) or H100, use **GKE** or **Compute Engine** instead
- Available regions: `us-east4`, `us-east4`, `europe-west1`, `europe-west4`, `asia-southeast1`

---

## Local Docker Testing

```bash
cp .env.example .env
# Edit .env with your HF_TOKEN and Twilio creds

docker compose up --build
# Server available at http://localhost:8080
```
