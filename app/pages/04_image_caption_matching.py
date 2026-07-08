from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from PIL import Image, UnidentifiedImageError
import streamlit as st

from app.config import MATCH_THRESHOLD
from app.components.display import (
    get_similarity_color,
    get_similarity_label,
    show_similarity_guide,
)
from app.components.footer import render_footer
from app.utils.data_loader import load_mini_clip_model, load_vocab
from app.utils.inference import encode_text_query, encode_uploaded_image


st.set_page_config(
    page_title="Image Caption Matching",
    page_icon="🔗",
    layout="wide",
)

st.title("🔗 Image + Caption Matching")

st.write(
    "Upload an image, enter a caption, and Mini-CLIP will estimate how well "
    "they match using cosine similarity."
)

st.subheader("📤 Upload an image and write a caption")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)

caption = st.text_input(
    "Enter a caption",
    placeholder="A dog running through a grassy field",
)

threshold = st.slider(
    "Match threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(MATCH_THRESHOLD),
    step=0.01,
)

st.divider()

if uploaded_file is not None and caption:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, RuntimeError, OSError) as error:
        st.error(
            "Unable to process this image. "
            "Please upload a valid JPG, JPEG, PNG, or WebP image."
        )

        with st.expander("Technical details"):
            st.code(str(error))

        st.stop()

    with st.spinner("Computing similarity..."):
        vocab = load_vocab()
        model = load_mini_clip_model(vocab_size=len(vocab))

        image_embedding = encode_uploaded_image(
            image=image,
            model=model,
        )

        text_embedding = encode_text_query(
            text=caption,
            model=model,
            vocab=vocab,
        )

        score = float((image_embedding * text_embedding).sum().item())

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded image")
        st.image(image, width="stretch")

    with col2:
        st.subheader("Matching result")

        label = get_similarity_label(score)
        color = get_similarity_color(score)

        is_match = score >= threshold
        decision = "✅ Match" if is_match else "❌ Not a match"

        st.markdown(f"### {decision}")

        st.markdown(f"**Caption:** {caption}")

        st.caption(label)

        st.markdown(
            f"""
            Similarity score:
            <span style="color:{color}; font-weight:700;">
                {score:.3f}
            </span>
            """,
            unsafe_allow_html=True,
        )

        st.progress(float(max(0.0, min(score, 1.0))))

        st.caption(f"Current threshold: {threshold:.2f}")

elif uploaded_file is None and not caption:
    st.info("Upload an image and enter a caption to compare them.")
elif uploaded_file is None:
    st.info("Upload an image to continue.")
else:
    st.info("Enter a caption to continue.")

st.divider()

with st.expander("About this demo"):
    st.write(
        """
        Mini-CLIP embeds the uploaded image and the entered caption into the same
        shared embedding space. Their cosine similarity is used as a matching score.

        The threshold converts the similarity score into a binary match decision.
        A threshold around 0.25 performed best during validation experiments, but
        you can adjust it here to make the matcher more or less strict.
        """
    )

show_similarity_guide()

render_footer()
