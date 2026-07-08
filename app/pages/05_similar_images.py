from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from PIL import Image, UnidentifiedImageError
import streamlit as st

from app.config import APP_IMAGES_DIR, TOP_K
from app.components.display import make_thumbnail
from app.components.footer import render_footer
from app.utils.inference import encode_uploaded_image
from app.utils.search import search_top_k

from app.utils.data_loader import (
    load_caption_index,
    load_image_index,
    load_mini_clip_model,
    load_vocab,
)
from app.utils.explanations import explain_image_similarity


st.set_page_config(
    page_title="Similar Images",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Similar Images")

st.write(
    "Upload an image and Mini-CLIP will retrieve visually or semantically similar "
    "images from the prepared Flickr30k image index."
)

st.subheader("📤 Upload an image")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

top_k = st.slider(
    "Number of similar images to retrieve",
    min_value=1,
    max_value=10,
    value=TOP_K,
)

st.divider()

if uploaded_file is not None:
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

    with st.spinner("Searching similar images..."):
        vocab = load_vocab()
        model = load_mini_clip_model(vocab_size=len(vocab))
        images_df, image_embeddings = load_image_index()
        captions_df, caption_embeddings = load_caption_index()

        query_embedding = encode_uploaded_image(
            image=image,
            model=model,
        )

        query_caption_results = search_top_k(
            query_embedding=query_embedding,
            candidate_embeddings=caption_embeddings,
            top_k=10,
        )

        query_captions = [
            captions_df.iloc[result["index"]]["caption"]
            for result in query_caption_results
        ]

        results = search_top_k(
            query_embedding=query_embedding,
            candidate_embeddings=image_embeddings,
            top_k=top_k,
        )

    st.subheader("Query image")

    left, center, right = st.columns([1, 2, 1])

    with center:
        st.image(image, width="stretch")

    st.divider()

    st.subheader("Retrieved images")

    cols = st.columns(min(top_k, 3))

    for rank, result in enumerate(results, start=1):
        image_row = images_df.iloc[result["index"]]
        image_path = APP_IMAGES_DIR / image_row["filename"]

        score = result["score"]

        with cols[(rank - 1) % len(cols)]:
            with st.container(border=True):
                result_image = Image.open(image_path)
                thumbnail = make_thumbnail(result_image)

                st.image(thumbnail, width="stretch")
                st.markdown(f"**{rank}. Similarity: `{score:.3f}`**")
                st.progress(float(max(0.0, min(score, 1.0))))

                image_idx = image_row["image_idx"]
                retrieved_captions = captions_df[
                    captions_df["image_idx"] == image_idx
                ]["caption"].tolist()

                explanation = explain_image_similarity(
                    query_captions=query_captions,
                    retrieved_captions=retrieved_captions,
                )

                with st.expander("Why this image?"):
                    st.markdown(explanation)
else:
    st.info("Upload an image to retrieve similar images.")

st.divider()

with st.expander("About this demo"):
    st.write(
        """
        The uploaded image is embedded into the Mini-CLIP shared representation
        space and compared against the prepared Flickr30k image index.

        Retrieved images may be visually similar, semantically similar, or both.
        For example, the model may retrieve images with similar objects, actions,
        settings, or overall scene structure.
        """
    )

render_footer()
