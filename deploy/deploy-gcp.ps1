# Production deployment script for Google Cloud Platform (PowerShell)

Write-Host "Starting GCP Cloud Run deployment for Agentic RAG Platform..." -ForegroundColor Cyan

# Check for gcloud CLI installation
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "Error: 'gcloud' CLI is not installed. Please install the Google Cloud SDK."
    exit 1
}

# Configure deployment parameters
$ProjectID = $env:GCP_PROJECT_ID
if (-not $ProjectID) {
    $ProjectID = (gcloud config get-value project 2>$null)
}
if (-not $ProjectID) {
    $ProjectID = Read-Host "Enter your GCP Project ID"
    gcloud config set project $ProjectID
}

$Region = "us-central1"
$ServiceName = "agentic-rag-service"
$RepoName = "agentic-rag-repo"
$ImageTag = "latest"
$ModelArmorEnabled = if ($env:MODEL_ARMOR_ENABLED) { $env:MODEL_ARMOR_ENABLED } else { "true" }
$ModelArmorProjectId = if ($env:MODEL_ARMOR_PROJECT_ID) { $env:MODEL_ARMOR_PROJECT_ID } else { "agentic-rag-504707" }
$ModelArmorLocation = if ($env:MODEL_ARMOR_LOCATION) { $env:MODEL_ARMOR_LOCATION } else { "us-central1" }
$ModelArmorTemplateId = if ($env:MODEL_ARMOR_TEMPLATE_ID) { $env:MODEL_ARMOR_TEMPLATE_ID } else { "my-rag-guardrail-template" }

Write-Host "GCP Project: $ProjectID" -ForegroundColor Green
Write-Host "Region: $Region" -ForegroundColor Green
Write-Host "Service: $ServiceName" -ForegroundColor Green

# Enable required Google Cloud service APIs
Write-Host "Enabling required GCP APIs..." -ForegroundColor Yellow
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com modelarmor.googleapis.com --project=$ProjectID

# Ensure Artifact Registry repository exists
Write-Host "Configuring Artifact Registry repository..." -ForegroundColor Yellow
gcloud artifacts repositories create $RepoName --repository-format=docker --location=$Region --description="Agentic RAG Docker repo" --project=$ProjectID 2>$null

$ImageUrl = "$Region-docker.pkg.dev/$ProjectID/$RepoName/agentic-rag:$ImageTag"

# Build and submit container image using Cloud Build
Write-Host "Building container image in Cloud Build..." -ForegroundColor Yellow
gcloud builds submit --tag $ImageUrl .

# Deploy container service to Cloud Run
Write-Host "Deploying service to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy $ServiceName `
    --image=$ImageUrl `
    --region=$Region `
    --platform=managed `
    --allow-unauthenticated `
    --memory=2Gi `
    --cpu=2 `
    --timeout=300s `
    --concurrency=80 `
    --min-instances=0 `
    --max-instances=10 `
    --set-secrets="GOOGLE_API_KEY=GOOGLE_API_KEY:latest,SECRET_KEY=SECRET_KEY:latest,TAVILY_API_KEY=TAVILY_API_KEY:latest,GROQ_API_KEY=GROQ_API_KEY:latest,QDRANT_API_KEY=QDRANT_API_KEY:latest,LANGSMITH_API_KEY=LANGSMITH_API_KEY:latest" `
    --set-env-vars="ENVIRONMENT=production,LLM_MODEL=gemini/gemini-3.1-flash-lite,EMBEDDING_MODEL=models/gemini-embedding-001,VECTOR_DB_TYPE=qdrant,QDRANT_COLLECTION_NAME=agentic_rag_documents,QDRANT_URL=https://7e28c9e6-5aa7-41b9-9a5e-49b50e5a650c.us-east4-0.gcp.cloud.qdrant.io,RETRIEVAL_TOP_K=5,USE_HYBRID_SEARCH=true,LANGSMITH_TRACING=true,LANGSMITH_ENDPOINT=https://api.smith.langchain.com,LANGSMITH_PROJECT=Agentic_RAG,MODEL_ARMOR_ENABLED=$ModelArmorEnabled,MODEL_ARMOR_PROJECT_ID=$ModelArmorProjectId,MODEL_ARMOR_LOCATION=$ModelArmorLocation,MODEL_ARMOR_TEMPLATE_ID=$ModelArmorTemplateId" `
    --project=$ProjectID

# Retrieve deployed service URL
$ServiceUrl = gcloud run services describe $ServiceName --platform=managed --region=$Region --format='value(status.url)'

Write-Host ""
Write-Host "Deployment completed successfully." -ForegroundColor Green
Write-Host "Service URL: $ServiceUrl" -ForegroundColor Cyan
Write-Host "API Documentation: $ServiceUrl/docs" -ForegroundColor Cyan
