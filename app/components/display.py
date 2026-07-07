import streamlit as st
from PIL import Image


def make_thumbnail(image, size=(320, 240)):
    """
    Resize an image while preserving its aspect ratio and place it on
    a fixed-size canvas for consistent display.
    """
    image = image.convert("RGB")
    thumbnail = Image.new("RGB", size, color=(245, 245, 245))

    image_copy = image.copy()
    image_copy.thumbnail(size)

    x = (size[0] - image_copy.width) // 2
    y = (size[1] - image_copy.height) // 2

    thumbnail.paste(image_copy, (x, y))

    return thumbnail


def show_similarity_guide():
    """
    Display a brief explanation of Mini-CLIP similarity scores.
    """
    with st.expander("How to interpret similarity scores"):
        st.markdown(
            """
            Similarity scores measure how close an image and caption are in the
            Mini-CLIP embedding space. They are **not probabilities**, but higher
            values usually indicate stronger semantic similarity.

            - 🟢 **0.67+**: strong semantic match
            - 🟡 **0.56–0.67**: good match
            - 🟠 **0.50–0.56**: moderate match
            - 🔴 **0.45–0.50**: weak match
            - ⚫ **below 0.45**: low similarity

            The model may perform better on clear objects and actions, and may be
            less reliable for unusual images, complex scenes, or concepts that are
            not well represented in the caption index.
            """
        )


def get_similarity_label(score):
    """
    Return a qualitative label for a cosine similarity score.
    """
    if score >= 0.75:
        return "🟢 Excellent match"
    if score >= 0.67:
        return "🟢 Strong match"
    if score >= 0.56:
        return "🟡 Good match"
    if score >= 0.50:
        return "🟠 Moderate match"
    if score >= 0.45:
        return "🔴 Weak match"

    return "⚫ Low similarity"


def get_similarity_color(score):
    """
    Return a display color corresponding to a similarity score.
    """
    if score >= 0.67:
        return "#2E8B57"  # green
    if score >= 0.50:
        return "#E6A700"  # amber
    return "#D32F2F"      # red
