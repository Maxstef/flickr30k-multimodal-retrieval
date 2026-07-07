import torch
import torch.nn.functional as F

from src.config import DEVICE
from src.features.image_features import (
    get_resnet18_feature_extractor,
    get_resnet18_transforms,
)


@torch.inference_mode()
def encode_uploaded_image(image, model):
    image_feature_extractor = get_resnet18_feature_extractor(DEVICE)
    image_transform = get_resnet18_transforms()

    image = image.convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    image_features = image_feature_extractor(image_tensor)

    image_embedding = model.image_projection(image_features)
    image_embedding = F.normalize(image_embedding, dim=1)

    return image_embedding.cpu()