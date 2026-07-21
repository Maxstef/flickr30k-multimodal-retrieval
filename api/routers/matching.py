"""Image-caption matching API routes."""

from io import BytesIO

import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from api.dependencies import get_model, get_vocab
from api.schemas import ImageCaptionMatchResponse
from src.serving.inference import encode_image, encode_text


MATCH_THRESHOLD = 0.25

router = APIRouter(
    prefix="/image-caption-match",
    tags=["Matching"],
)


@router.post(
    "",
    response_model=ImageCaptionMatchResponse,
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
    normalized_caption = caption.strip()

    if not normalized_caption:
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
        pil_image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from exc

    image_embedding = encode_image(
        image=pil_image,
        model=model,
    )

    text_embedding = encode_text(
        text=normalized_caption,
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
