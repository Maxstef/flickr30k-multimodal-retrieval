"""Pydantic schemas for retrieval endpoints."""

from pydantic import BaseModel, Field


class CaptionResult(BaseModel):
    """
    A caption returned by an image-to-caption retrieval request.
    """

    caption: str = Field(
        description="Retrieved caption.",
    )
    score: float = Field(
        description="Cosine similarity between the image and caption embeddings.",
    )


class ImageResult(BaseModel):
    """
    An image returned by a caption-to-image retrieval request.
    """

    image_id: int = Field(
        description="Identifier of the retrieved image.",
    )
    filename: str = Field(
        description="Filename of the retrieved image.",
    )
    score: float = Field(
        description="Cosine similarity between the caption and image embeddings.",
    )


class ImageToCaptionResponse(BaseModel):
    """
    Response returned by the image-to-caption retrieval endpoint.
    """

    top_k: int = Field(
        description="Number of requested retrieval results.",
    )
    results: list[CaptionResult] = Field(
        description="Captions ranked by similarity to the uploaded image.",
    )


class CaptionToImageResponse(BaseModel):
    """
    Response returned by the caption-to-image retrieval endpoint.
    """

    query: str = Field(
        description="Caption used as the retrieval query.",
    )
    top_k: int = Field(
        description="Number of requested retrieval results.",
    )
    results: list[ImageResult] = Field(
        description="Images ranked by similarity to the query caption.",
    )