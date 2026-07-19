from fastapi import FastAPI

app = FastAPI(
    title="Mini-CLIP API",
    description=(
        "REST API for multimodal image-text retrieval and matching "
        "using the Mini-CLIP model."
    ),
    version="1.0.0",
)


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    """Return basic API information."""
    return {
        "name": "Mini-CLIP API",
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/health", tags=["General"])
def health_check() -> dict[str, str]:
    """Check whether the API service is running."""
    return {"status": "healthy"}