"""FastAPI application entry point."""

from fastapi import FastAPI

from api.routers.matching import router as matching_router
from api.routers.retrieval import router as retrieval_router


app = FastAPI(
    title="Mini-CLIP API",
    description=(
        "REST API for multimodal image-text retrieval and matching "
        "using the Mini-CLIP model."
    ),
    version="1.0.0",
)

app.include_router(matching_router)
app.include_router(retrieval_router)


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