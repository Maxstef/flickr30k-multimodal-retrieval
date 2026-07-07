import torch
import torch.nn.functional as F

from src.features.image_features import (
    get_resnet18_feature_extractor,
    get_resnet18_transforms,
)
from src.text.tokenization import encode_caption
from src.config import (
    DEVICE,
    MAX_CAPTION_LENGTH,
)


@torch.inference_mode()
def encode_uploaded_image(image, model):
    """
    Encode an uploaded image into the shared Mini-CLIP embedding space.
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
def encode_text_query(text, model, vocab):
    """
    Encode a text query into the shared Mini-CLIP embedding space.
    """
    caption_ids = encode_caption(
        text,
        vocab,
        max_length=MAX_CAPTION_LENGTH,
    )

    caption_tensor = torch.tensor(
        caption_ids,
        dtype=torch.long,
    ).unsqueeze(0).to(DEVICE)

    # The Mini-CLIP forward pass expects both image and text inputs.
    # A dummy image feature tensor is used because only the text embedding
    # is required for retrieval.
    dummy_image_features = torch.zeros(
        1,
        model.image_projection.in_features,
        device=DEVICE,
    )

    _, text_embedding = model(dummy_image_features, caption_tensor)

    return text_embedding.cpu()
