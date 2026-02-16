#!/usr/bin/env bash
# =============================================================================
# PersonaPlex -- Full GCP Cloud Run (GPU) Deployment Script
#
# This script automates the ENTIRE deployment pipeline:
#   1. Validates prerequisites (gcloud CLI, APIs)
#   2. Creates Artifact Registry repo
#   3. Creates GCP secrets (Twilio creds -- HF token is assumed pre-configured)
#   4. Builds and pushes the Docker image (via Cloud Build -- no local Docker needed)
#   5. Deploys to Cloud Run with L4 GPU
#   6. Sets SERVER_URL back on the service
#   7. Prints Twilio webhook configuration instructions
#
# Usage:
#   chmod +x deploy/setup-gcp.sh
#   ./deploy/setup-gcp.sh
#
# Required env vars (set before running OR the script will prompt):
#   TWILIO_ACCOUNT_SID   -- Your Twilio account SID
#   TWILIO_AUTH_TOKEN     -- Your Twilio auth token
#   TWILIO_PHONE_NUMBER   -- Your Twilio phone number (E.164 format)
#
# Optional env vars:
#   GCP_PROJECT_ID        -- GCP project (defaults to current gcloud project)
#   GCP_REGION            -- Region (default: us-east4)
#   SERVICE_NAME          -- Cloud Run service name (default: personaplex)
#   MIN_INSTANCES         -- Minimum instances, 0 = scale-to-zero (default: 0)
#   MAX_INSTANCES         -- Maximum instances (default: 1)
#   HF_SECRET_NAME        -- Secret Manager name for HF token (default: hf-token)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Colors for output
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }
step()  { echo -e "\n${GREEN}==> Step $1: $2${NC}"; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
GCP_PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || echo "")}"
GCP_REGION="${GCP_REGION:-us-east4}"
SERVICE_NAME="${SERVICE_NAME:-personaplex}"
AR_REPO="${SERVICE_NAME}"
IMAGE_NAME="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${SERVICE_NAME}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"
HF_SECRET_NAME="${HF_SECRET_NAME:-hf-token}"

# ---------------------------------------------------------------------------
# Step 0: Validate prerequisites
# ---------------------------------------------------------------------------
step 0 "Validating prerequisites"

if ! command -v gcloud &>/dev/null; then
    err "gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi
ok "gcloud CLI found"

if [ -z "$GCP_PROJECT_ID" ]; then
    err "No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi
ok "GCP project: $GCP_PROJECT_ID"
info "Region: $GCP_REGION"
info "Service: $SERVICE_NAME"

# ---------------------------------------------------------------------------
# Step 1: Enable required APIs
# ---------------------------------------------------------------------------
step 1 "Enabling GCP APIs"

APIS=(
    "run.googleapis.com"
    "artifactregistry.googleapis.com"
    "secretmanager.googleapis.com"
    "cloudbuild.googleapis.com"
)
for api in "${APIS[@]}"; do
    info "Enabling $api..."
    gcloud services enable "$api" --project="$GCP_PROJECT_ID" --quiet
done
ok "All APIs enabled"

# ---------------------------------------------------------------------------
# Step 2: Create Artifact Registry repository
# ---------------------------------------------------------------------------
step 2 "Setting up Artifact Registry"

if gcloud artifacts repositories describe "$AR_REPO" \
    --location="$GCP_REGION" --project="$GCP_PROJECT_ID" &>/dev/null; then
    ok "Artifact Registry repo '$AR_REPO' already exists"
else
    info "Creating Artifact Registry repo '$AR_REPO'..."
    gcloud artifacts repositories create "$AR_REPO" \
        --repository-format=docker \
        --location="$GCP_REGION" \
        --project="$GCP_PROJECT_ID" \
        --description="PersonaPlex container images"
    ok "Artifact Registry repo created"
fi

ok "Artifact Registry ready (Cloud Build handles auth automatically)"

# ---------------------------------------------------------------------------
# Step 3: Verify HF token secret exists
# ---------------------------------------------------------------------------
step 3 "Verifying HF token in Secret Manager"

if gcloud secrets describe "$HF_SECRET_NAME" --project="$GCP_PROJECT_ID" &>/dev/null; then
    ok "Secret '$HF_SECRET_NAME' found in Secret Manager"
else
    err "Secret '$HF_SECRET_NAME' NOT found in Secret Manager."
    echo ""
    echo "  Your HF token must be stored as a secret named '$HF_SECRET_NAME'."
    echo "  If yours has a different name, re-run with: HF_SECRET_NAME=your-name ./deploy/setup-gcp.sh"
    echo ""
    echo "  To create it:"
    echo "    echo -n 'hf_YOUR_TOKEN' | gcloud secrets create $HF_SECRET_NAME --data-file=- --project=$GCP_PROJECT_ID"
    echo ""
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 4: Create Twilio secrets
# ---------------------------------------------------------------------------
step 4 "Setting up Twilio secrets"

# Prompt for Twilio creds if not set
if [ -z "${TWILIO_ACCOUNT_SID:-}" ]; then
    read -rp "Enter TWILIO_ACCOUNT_SID: " TWILIO_ACCOUNT_SID
fi
if [ -z "${TWILIO_AUTH_TOKEN:-}" ]; then
    read -rsp "Enter TWILIO_AUTH_TOKEN: " TWILIO_AUTH_TOKEN
    echo ""
fi
if [ -z "${TWILIO_PHONE_NUMBER:-}" ]; then
    read -rp "Enter TWILIO_PHONE_NUMBER (E.164, e.g. +14155551234): " TWILIO_PHONE_NUMBER
fi

create_or_update_secret() {
    local name="$1"
    local value="$2"
    if gcloud secrets describe "$name" --project="$GCP_PROJECT_ID" &>/dev/null; then
        info "Updating secret '$name'..."
        echo -n "$value" | gcloud secrets versions add "$name" --data-file=- --project="$GCP_PROJECT_ID"
    else
        info "Creating secret '$name'..."
        echo -n "$value" | gcloud secrets create "$name" --data-file=- --project="$GCP_PROJECT_ID"
    fi
}

create_or_update_secret "twilio-account-sid" "$TWILIO_ACCOUNT_SID"
create_or_update_secret "twilio-auth-token" "$TWILIO_AUTH_TOKEN"
create_or_update_secret "twilio-phone-number" "$TWILIO_PHONE_NUMBER"
ok "Twilio secrets configured"

# ---------------------------------------------------------------------------
# Step 5: Grant Cloud Run access to secrets
# ---------------------------------------------------------------------------
step 5 "Granting Cloud Run service account access to secrets"

PROJECT_NUMBER=$(gcloud projects describe "$GCP_PROJECT_ID" --format='value(projectNumber)')
SA_EMAIL="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

SECRETS=("$HF_SECRET_NAME" "twilio-account-sid" "twilio-auth-token" "twilio-phone-number")
for secret in "${SECRETS[@]}"; do
    gcloud secrets add-iam-policy-binding "$secret" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/secretmanager.secretAccessor" \
        --project="$GCP_PROJECT_ID" --quiet 2>/dev/null || true
done
ok "Secret access granted to service account"

# ---------------------------------------------------------------------------
# Step 6: Build and push Docker image (via Cloud Build -- no local Docker needed)
# ---------------------------------------------------------------------------
step 6 "Building Docker image with Cloud Build"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

info "Submitting build to Cloud Build..."
info "Image: ${IMAGE_NAME}:latest"
info "Build context: ${REPO_ROOT}"

gcloud builds submit "$REPO_ROOT" \
    --tag="${IMAGE_NAME}:latest" \
    --project="$GCP_PROJECT_ID" \
    --machine-type=e2-highcpu-8 \
    --timeout=1800 \
    --quiet

ok "Image built and pushed: ${IMAGE_NAME}:latest"

# ---------------------------------------------------------------------------
# Step 7: Deploy to Cloud Run
# ---------------------------------------------------------------------------
step 7 "Deploying to Cloud Run with L4 GPU"

info "Deploying ${SERVICE_NAME}..."
gcloud run deploy "$SERVICE_NAME" \
    --image="${IMAGE_NAME}:latest" \
    --region="$GCP_REGION" \
    --project="$GCP_PROJECT_ID" \
    --execution-environment=gen2 \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --cpu=8 \
    --memory=32Gi \
    --concurrency=1 \
    --min-instances="$MIN_INSTANCES" \
    --max-instances="$MAX_INSTANCES" \
    --timeout=3600 \
    --port=8080 \
    --no-cpu-throttling \
    --set-env-vars="DEVICE=cuda,NO_TORCH_COMPILE=1,HF_REPO=nvidia/personaplex-7b-v1,HF_HUB_ENABLE_HF_TRANSFER=1" \
    --set-secrets="HF_TOKEN=${HF_SECRET_NAME}:latest,TWILIO_ACCOUNT_SID=twilio-account-sid:latest,TWILIO_AUTH_TOKEN=twilio-auth-token:latest,TWILIO_PHONE_NUMBER=twilio-phone-number:latest" \
    --allow-unauthenticated \
    --quiet

ok "Cloud Run service deployed"

# ---------------------------------------------------------------------------
# Step 8: Get service URL and set SERVER_URL
# ---------------------------------------------------------------------------
step 8 "Configuring service URL"

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region="$GCP_REGION" \
    --project="$GCP_PROJECT_ID" \
    --format='value(status.url)')

info "Service URL: $SERVICE_URL"

gcloud run services update "$SERVICE_NAME" \
    --region="$GCP_REGION" \
    --project="$GCP_PROJECT_ID" \
    --update-env-vars="SERVER_URL=${SERVICE_URL}" \
    --quiet

ok "SERVER_URL set to ${SERVICE_URL}"

# ---------------------------------------------------------------------------
# Step 9: Wait for healthy startup
# ---------------------------------------------------------------------------
step 9 "Waiting for service to become healthy"

info "The model takes ~90-120 seconds to load. Polling /health..."
MAX_WAIT=360
ELAPSED=0
INTERVAL=15

while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health" 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        ok "Service is healthy!"
        curl -s "${SERVICE_URL}/health" | python3 -m json.tool 2>/dev/null || true
        break
    fi
    info "Status: $STATUS (elapsed: ${ELAPSED}s / ${MAX_WAIT}s)"
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    warn "Service not healthy after ${MAX_WAIT}s. Check logs:"
    echo "  gcloud run logs read $SERVICE_NAME --region=$GCP_REGION --project=$GCP_PROJECT_ID --limit=50"
fi

# ---------------------------------------------------------------------------
# Step 10: Print summary
# ---------------------------------------------------------------------------
step 10 "Deployment complete"

echo ""
echo "============================================================================="
echo "  PersonaPlex Deployment Summary"
echo "============================================================================="
echo ""
echo "  Service URL:     ${SERVICE_URL}"
echo "  Health check:    ${SERVICE_URL}/health"
echo "  API docs:        ${SERVICE_URL}/docs"
echo "  React UI:        ${SERVICE_URL}/"
echo ""
echo "  Twilio Endpoints:"
echo "    Inbound hook:  ${SERVICE_URL}/twilio/inbound"
echo "    Outbound API:  ${SERVICE_URL}/twilio/outbound"
echo "    Media stream:  wss://$(echo "$SERVICE_URL" | sed 's|https://||')/twilio/media-stream"
echo ""
echo "============================================================================="
echo "  MANUAL STEP REQUIRED -- Configure Twilio Webhook"
echo "============================================================================="
echo ""
echo "  1. Go to: https://console.twilio.com/us1/develop/phone-numbers/manage/incoming"
echo "  2. Click your phone number: ${TWILIO_PHONE_NUMBER}"
echo "  3. Under 'Voice Configuration':"
echo "     - Set 'A call comes in' to: Webhook"
echo "     - URL:    ${SERVICE_URL}/twilio/inbound"
echo "     - Method: HTTP POST"
echo "  4. Click 'Save configuration'"
echo ""
echo "============================================================================="
echo "  Useful Commands"
echo "============================================================================="
echo ""
echo "  View logs:"
echo "    gcloud run logs read $SERVICE_NAME --region=$GCP_REGION --project=$GCP_PROJECT_ID --limit=50"
echo ""
echo "  Redeploy after code changes:"
echo "    gcloud builds submit . --tag=${IMAGE_NAME}:latest --project=$GCP_PROJECT_ID"
echo "    gcloud run deploy $SERVICE_NAME --image=${IMAGE_NAME}:latest --region=$GCP_REGION --project=$GCP_PROJECT_ID"
echo ""
echo "  Make an outbound test call:"
echo "    curl -X POST '${SERVICE_URL}/twilio/outbound?to=+1YOURNUMBER&voice_prompt=NATF2.pt&text_prompt=Hello'"
echo ""
echo "  Scale to zero (save costs when idle):"
echo "    gcloud run services update $SERVICE_NAME --region=$GCP_REGION --min-instances=0"
echo ""
echo "  Delete everything:"
echo "    gcloud run services delete $SERVICE_NAME --region=$GCP_REGION"
echo "    gcloud artifacts repositories delete $AR_REPO --location=$GCP_REGION"
echo ""
