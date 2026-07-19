"""Framework-independent Mini-CLIP inference utilities."""

from PIL import Image
import torch
import torch.nn.functional as F

from src.config import DEVICE, MAX_CAPTION_LENGTH
from src.features.image_features import (
    get_resnet18_feature_extractor,
    get_resnet18_transforms,
)
from src.models.mini_clip import MiniCLIP
from src.text.tokenization import encode_caption


@torch.inference_mode()
def encode_image(
    image: Image.Image,
    model: MiniCLIP,
) -> torch.Tensor:
    """
    Encode an image into the shared Mini-CLIP embedding space.

    Args:
        image:
            PIL image to encode.
        model:
            Trained Mini-CLIP model.

    Returns:
        torch.Tensor:
            Normalized image embedding with shape ``[1, projection_dim]``
            stored on the CPU.
    """
    image_feature_extractor = get_resnet18_feature_extractor(DEVICE)
    image_transform = get_resnet18_transforms()

    image = image.convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    image_features = image_feature_extractor(image_tensor)
    image_embedding = model.image_projection(image_features)
    image_embedding = F.normalize(image_embedding, dim=1)

    return image_embedding.cpu()


@torch.inference_mode()
def encode_text(
    text: str,
    model: MiniCLIP,
    vocab: dict[str, int],
) -> torch.Tensor:
    """
    Encode text into the shared Mini-CLIP embedding space.

    Args:
        text:
            Caption or search query to encode.
        model:
            Trained Mini-CLIP model.
        vocab:
            Vocabulary used by the Mini-CLIP text encoder.

    Returns:
        torch.Tensor:
            Text embedding with shape ``[1, projection_dim]``
            stored on the CPU.
    """
    caption_ids = encode_caption(
        text,
        vocab,
        max_length=MAX_CAPTION_LENGTH,
    )

    caption_tensor = torch.tensor(
        caption_ids,
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    dummy_image_features = torch.zeros(
        1,
        model.image_projection.in_features,
        device=DEVICE,
    )

    _, text_embedding = model(
        dummy_image_features,
        caption_tensor,
    )

    return text_embedding.cpu()