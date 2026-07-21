"""Routes for serving indexed image files."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import APP_IMAGES_DIR as IMAGE_DIR


router = APIRouter(
    prefix="/images",
    tags=["images"],
)


@router.get(
    "/{filename}",
    name="get_image",
)
async def get_image(filename: str) -> FileResponse:
    """
    Download an indexed image by filename.
    """
    image_directory = Path(IMAGE_DIR).resolve()
    image_path = (image_directory / filename).resolve()

    try:
        image_path.relative_to(image_directory)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid image filename.",
        ) from error

    if not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Image not found.",
        )

    return FileResponse(
        path=image_path,
        filename=image_path.name,
    )