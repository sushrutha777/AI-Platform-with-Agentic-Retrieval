# Google Cloud Platform (GCP) Deployment Guide

This guide details how to deploy the **Enterprise AI Platform with Agentic Retrieval** to **Google Cloud Platform (GCP)** using **Cloud Run**, **Artifact Registry**, and **Secret Manager**.

---

## GCP Architecture Overview

```
                          Internet (HTTPS / Custom Domain)
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │      Google Cloud Run       │
                         │ (Auto-scale 0..N Instances) │
                         ├─────────────────────────────┤
                         │  • React SPA Frontend       │
                         │  • FastAPI Backend          │
                         │  • LangGraph Reasoning      │
                         │  • SSE Real-time Streaming  │
                         └──────────────┬──────────────┘
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 ▼                      ▼                      ▼
        ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
        │ Secret Manager  │    │  Google Gemini  │    │  Tavily Search  │
        │ • GOOGLE_API_KEY│    │  2.0 Flash API  │    │  Wikipedia API  │
        │ • SECRET_KEY    │    └─────────────────┘    └─────────────────┘
        └─────────────────┘
```

---

## Application request-boundary architecture

```text
React
  ↓
FastAPI
  ↓
Model Armor: user-prompt sanitization
  ↓
Existing heuristic gatekeeper and zero-LLM router
  ↓
Parallel Qdrant/BM25/web/Wikipedia retrieval
  ↓
RRF and FlashRank reranking
  ↓
Single final LiteLLM call
  ↓
Gemini primary or Groq fallback
  ↓
Model Armor: model-response sanitization
  ↓
FastAPI / React
```

Model Armor is applied at the application boundary. It is not placed between Gemini and Groq, and it does not add another LLM call or screen each retrieved result individually.

---

## Prerequisites

1. **Google Cloud SDK (`gcloud`)** installed:
   ```bash
   gcloud version
   ```
2. **Google Cloud Account** with an active billing project.
3. Authenticate with your GCP account:
   ```bash
   gcloud auth login
   gcloud auth configure-docker
   ```

---

## Model Armor configuration

The application uses the existing Model Armor template configured through runtime environment variables. It does not create or update templates. Enable the API in the project that hosts the template:

```bash
gcloud services enable modelarmor.googleapis.com --project=agentic-rag-504707
```

The Cloud Run service account is not specified in this repository's deployment scripts, so resolve the identity from the deployed service before granting access:

```bash
SERVICE_ACCOUNT=$(gcloud run services describe SERVICE_NAME \
  --region=SERVICE_REGION \
  --project=RUN_PROJECT_ID \
  --format='value(spec.template.spec.serviceAccountName)')

gcloud projects add-iam-policy-binding agentic-rag-504707 \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/modelarmor.user"
```

`roles/modelarmor.user` is the least-privilege predefined role for an application that uses existing templates for user-prompt and model-response sanitization. If the Cloud Run service runs in another project, grant this role in the template-hosting project (`agentic-rag-504707`).

Set these non-secret variables on Cloud Run; no Model Armor key or service-account JSON file is required:

```text
MODEL_ARMOR_ENABLED=true
MODEL_ARMOR_PROJECT_ID=agentic-rag-504707
MODEL_ARMOR_LOCATION=us-central1
MODEL_ARMOR_TEMPLATE_ID=my-rag-guardrail-template
MODEL_ARMOR_TIMEOUT_SECONDS=5
```

---

## Method 1: Automated Script Deployment

Preconfigured deployment scripts are provided:

### On Windows (PowerShell):
```powershell
.\deploy\deploy-gcp.ps1
```

### On Linux / macOS / Cloud Shell:
```bash
chmod +x ./deploy/deploy-gcp.sh
./deploy/deploy-gcp.sh
```

The script performs the following tasks:
1. Enables `run.googleapis.com`, `artifactregistry.googleapis.com`, `cloudbuild.googleapis.com`, and `modelarmor.googleapis.com`.
2. Builds the unified production container (React + FastAPI) using Google Cloud Build.
3. Deploys the container to Cloud Run with 2 vCPUs, 2 GB RAM, and auto-scaling.
4. Outputs the live HTTPS application URL.

---

## Method 2: Step-by-Step Manual Deployment

### Step 1: Set Project & Enable APIs
```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"

gcloud config set project $PROJECT_ID

gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    secretmanager.googleapis.com \
    modelarmor.googleapis.com
```

### Step 2: Store Secrets in Secret Manager
```bash
# Google Gemini API Key
echo -n "AIzaSy..." | gcloud secrets create GOOGLE_API_KEY --data-file=-

# JWT Secret Key
echo -n "$(openssl rand -hex 32)" | gcloud secrets create SECRET_KEY --data-file=-

# Optional: Tavily Search API Key
echo -n "tvly-..." | gcloud secrets create TAVILY_API_KEY --data-file=-
```

### Step 3: Create Artifact Registry Repository
```bash
gcloud artifacts repositories create agentic-rag-repo \
    --repository-format=docker \
    --location=$REGION \
    --description="Agentic RAG Docker repository"
```

### Step 4: Build & Push Image with Cloud Build
```bash
gcloud builds submit \
    --tag "$REGION-docker.pkg.dev/$PROJECT_ID/agentic-rag-repo/agentic-rag:latest" .
```

### Step 5: Deploy to Cloud Run
```bash
gcloud run deploy agentic-rag-service \
    --image="$REGION-docker.pkg.dev/$PROJECT_ID/agentic-rag-repo/agentic-rag:latest" \
    --region="$REGION" \
    --platform=managed \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=2 \
    --timeout=300s \
    --concurrency=80 \
    --min-instances=0 \
    --max-instances=10 \
    --set-secrets="GOOGLE_API_KEY=GOOGLE_API_KEY:latest,SECRET_KEY=SECRET_KEY:latest,TAVILY_API_KEY=TAVILY_API_KEY:latest" \
    --set-env-vars="ENVIRONMENT=production,LLM_MODEL=gemini-2.0-flash,EMBEDDING_MODEL=models/text-embedding-004,MODEL_ARMOR_ENABLED=true,MODEL_ARMOR_PROJECT_ID=agentic-rag-504707,MODEL_ARMOR_LOCATION=us-central1,MODEL_ARMOR_TEMPLATE_ID=my-rag-guardrail-template,MODEL_ARMOR_TIMEOUT_SECONDS=5"
```

---

## Method 3: Continuous Deployment with GitHub Actions

Add the following GitHub Actions workflow to `.github/workflows/deploy-gcp.yml`:

```yaml
name: Deploy to GCP Cloud Run

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Build & Submit via Cloud Build
        run: |
          gcloud builds submit --config=cloudbuild.yaml \
            --substitutions=_LOCATION=us-central1,_REPOSITORY=agentic-rag-repo
```

---

## Production Best Practices & Cost Optimization

| Feature | Configuration | Benefit |
| :--- | :--- | :--- |
| **Scale to Zero** | `--min-instances=0` | Zero infrastructure cost when idle; billing applies only during request execution. |
| **Concurrency** | `--concurrency=80` | Allows a single container instance to serve multiple concurrent SSE chat streams. |
| **Timeout** | `--timeout=300s` | Ensures extended multi-tool reasoning sequences complete without connection drops. |
| **Custom Domain** | Cloud Run Custom Domains | Automated, managed TLS/SSL certificate issuance and renewal. |
| **Health Check** | `/api/v1/health` | Standardized readiness and liveness probe endpoint. |
