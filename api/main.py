from io import BytesIO

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
import torch

from api.dependencies import get_model, get_vocab
from api.schemas import ImageCaptionMatchResponse
from src.serving.inference import encode_image, encode_text


MATCH_THRESHOLD = 0.25

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


@app.post(
    "/image-caption-match",
    response_model=ImageCaptionMatchResponse,
    tags=["Matching"],
)
async def image_caption_match(
    image: UploadFile = File(...),
    caption: str = Form(...),
    model=Depends(get_model),
    vocab: dict = Depends(get_vocab),
) -> ImageCaptionMatchResponse:
    """
    Calculate the similarity between an uploaded image and a caption.
    """
    if not caption.strip():
        raise HTTPException(
            status_code=400,
            detail="Caption must not be empty.",
        )

    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes))
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from exc

    image_embedding = encode_image(
        image=pil_image,
        model=model,
    )

    text_embedding = encode_text(
        text=caption.strip(),
        model=model,
        vocab=vocab,
    )

    similarity = torch.sum(
        image_embedding * text_embedding,
        dim=1,
    ).item()

    return ImageCaptionMatchResponse(
        similarity=similarity,
        threshold=MATCH_THRESHOLD,
        is_match=similarity >= MATCH_THRESHOLD,
    )