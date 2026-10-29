from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from ..config import settings

app = FastAPI(
    title="VeritasAI API",
    description="Enterprise Retrieval-Augmented Generation (RAG) Engine with Inline Citations and Strict Grounding",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for Kubernetes liveness probes."""
    return {
        "status": "HEALTHY",
        "service": "veritas-ai-rag",
        "environment": settings.ENVIRONMENT,
        "version": "1.0.0",
    }
