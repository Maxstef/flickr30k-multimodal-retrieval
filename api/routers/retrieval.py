"""Multimodal retrieval API routes."""

from io import BytesIO

import pandas as pd
import torch
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from PIL import Image, UnidentifiedImageError

from api.dependencies import (
    get_caption_index,
    get_image_index,
    get_model,
    get_vocab,
)
from api.schemas import (
    CaptionResult,
    CaptionToImageResponse,
    ImageResult,
    ImageToCaptionResponse,
)
from src.serving.inference import encode_image, encode_text
from src.serving.retrieval import retrieve_top_k

router = APIRouter(
    tags=["Retrieval"],
)


@router.post(
    "/image-to-caption",
    response_model=ImageToCaptionResponse,
)
async def image_to_caption(
    image: UploadFile = File(...),
    top_k: int = Query(
        default=3,
        ge=1,
        le=20,
        description="Number of captions to retrieve.",
    ),
    model=Depends(get_model),
    caption_index: tuple[pd.DataFrame, torch.Tensor] = Depends(
        get_caption_index
    ),
) -> ImageToCaptionResponse:
    """
    Retrieve the captions most similar to an uploaded image.
    """
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

    captions_df, caption_embeddings = caption_index

    if captions_df.empty or caption_embeddings.shape[0] == 0:
        raise HTTPException(
            status_code=503,
            detail="Caption retrieval index is empty.",
        )

    if "caption" not in captions_df.columns:
        raise HTTPException(
            status_code=500,
            detail="Caption metadata does not contain a 'caption' column.",
        )

    effective_top_k = min(
        top_k,
        len(captions_df),
        caption_embeddings.shape[0],
    )

    image_embedding = encode_image(
        image=pil_image,
        model=model,
    )

    retrieved_items = retrieve_top_k(
        query_embedding=image_embedding,
        candidate_embeddings=caption_embeddings,
        top_k=effective_top_k,
    )

    results = [
        CaptionResult(
            caption=str(
                captions_df.iloc[item["index"]]["caption"]
            ),
            score=item["score"],
        )
        for item in retrieved_items
    ]

    return ImageToCaptionResponse(
        top_k=effective_top_k,
        results=results,
    )

@router.post(
    "/caption-to-image",
    response_model=CaptionToImageResponse,
)
async def caption_to_image(
    request: Request,
    caption: str = Form(...),
    top_k: int = Query(
        default=3,
        ge=1,
        le=20,
        description="Number of images to retrieve.",
    ),
    model=Depends(get_model),
    vocab: dict = Depends(get_vocab),
    image_index: tuple[pd.DataFrame, torch.Tensor] = Depends(
        get_image_index
    ),
) -> CaptionToImageResponse:
    """
    Retrieve the images most similar to a caption.
    """
    normalized_caption = caption.strip()

    if not normalized_caption:
        raise HTTPException(
            status_code=400,
            detail="Caption must not be empty.",
        )

    images_df, image_embeddings = image_index

    if images_df.empty or image_embeddings.shape[0] == 0:
        raise HTTPException(
            status_code=503,
            detail="Image retrieval index is empty.",
        )

    required_columns = {"image_idx", "filename"}
    missing_columns = required_columns.difference(images_df.columns)

    if missing_columns:
        missing_columns_text = ", ".join(sorted(missing_columns))

        raise HTTPException(
            status_code=500,
            detail=(
                "Image metadata is missing required columns: "
                f"{missing_columns_text}."
            ),
        )

    effective_top_k = min(
        top_k,
        len(images_df),
        image_embeddings.shape[0],
    )

    text_embedding = encode_text(
        text=normalized_caption,
        model=model,
        vocab=vocab,
    )

    retrieved_items = retrieve_top_k(
        query_embedding=text_embedding,
        candidate_embeddings=image_embeddings,
        top_k=effective_top_k,
    )

    results = []

    for item in retrieved_items:
        image_row = images_df.iloc[item["index"]]
        filename = str(image_row["filename"])

        results.append(
            ImageResult(
                image_id=int(image_row["image_idx"]),
                filename=filename,
                image_url=str(
                    request.url_for(
                        "get_image",
                        filename=filename,
                    )
                ),
                score=item["score"],
            )
        )

    return CaptionToImageResponse(
        query=normalized_caption,
        top_k=effective_top_k,
        results=results,
    )