"""Pydantic schemas for image-caption matching endpoints."""

from pydantic import BaseModel, Field


class ImageCaptionMatchResponse(BaseModel):
    """
    Response returned by the image-caption matching endpoint.
    """

    similarity: float = Field(
        description="Cosine similarity between the image and text embeddings.",
    )
    threshold: float = Field(
        description="Similarity threshold used to determine the match.",
    )
    is_match: bool = Field(
        description="Whether the image and caption are considered a match.",
    )
