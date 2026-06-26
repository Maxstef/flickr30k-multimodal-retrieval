import torch
from torchvision import models
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix


def get_resnet18_feature_extractor(device):
    """
    Create a pretrained ResNet18 feature extractor.

    The final classification layer is removed, so the model outputs
    a 512-dimensional image feature vector.

    Args:
        device: Device used for inference.

    Returns:
        A ResNet18 model without its classification head.
    """

    # Load pretrained ImageNet weights
    weights = models.ResNet18_Weights.DEFAULT

    # Initialize ResNet18 with pretrained weights
    model = models.resnet18(weights=weights)

    # Remove the final classification layer.
    # Instead of predicting ImageNet classes, the model will output
    # the 512-dimensional feature vector produced by the last layer.
    model.fc = torch.nn.Identity()

    # Move model to CPU or GPU
    model = model.to(device)

    # Switch to evaluation mode by default
    model.eval()

    return model


def get_resnet18_transforms():
    """
    Get image preprocessing transforms required by pretrained ResNet18.

    Returns:
        torchvision transforms for image resizing, tensor conversion,
        and ImageNet normalization.
    """

    # Load the preprocessing pipeline associated with the pretrained weights.
    # This ensures images are transformed exactly as during ImageNet training.
    weights = models.ResNet18_Weights.DEFAULT

    return weights.transforms()


def extract_image_features(dataset, model, transform, device):
    """
    Extract image features for all images in a Flickr30k split.

    Args:
        dataset: Hugging Face Dataset split.
        model: Image feature extractor.
        transform: Image preprocessing transform.
        device: Device used for inference.

    Returns:
        Dictionary mapping filename to image feature tensor.
    """
    features = {}

    # Read filenames once for efficient column access
    filenames = dataset["filename"]

    # Iterate through every image in the dataset
    for idx in tqdm(range(len(dataset))):

        # Convert image to RGB because pretrained ResNet expects
        # three-channel RGB input.
        image = dataset[idx]["image"].convert("RGB")
        filename = filenames[idx]

        # Apply preprocessing:
        # - resize
        # - convert to tensor
        # - normalize using ImageNet statistics
        image_tensor = transform(image)

        # Add batch dimension:
        # (3, 224, 224) -> (1, 3, 224, 224)
        image_tensor = image_tensor.unsqueeze(0)

        # Move tensor to CPU or GPU
        image_tensor = image_tensor.to(device)

         # Disable gradient computation because we only perform inference.
        with torch.inference_mode():

            # Extract the 512-dimensional image embedding
            feature = model(image_tensor)

            # Remove the artificial batch dimension:
            # (1, 512) -> (512,)
            feature = feature.squeeze(0)

            # Move features back to CPU for storage
            feature = feature.cpu()

        # Store embedding using filename as the key
        features[filename] = feature

    return features


def build_image_feature_matrix(pairs, image_features):
    """
    Build an image feature matrix for image-caption pairs.

    Each row corresponds to the image embedding associated with the
    filename in the provided pairs DataFrame.

    Args:
        pairs: DataFrame containing image-caption pairs.
        image_features: Dictionary mapping filenames to image embeddings.

    Returns:
        A sparse matrix of image features.
    """
    features = [
        image_features[filename].cpu().numpy()
        for filename in pairs["filename"]
    ]

    return csr_matrix(np.vstack(features))