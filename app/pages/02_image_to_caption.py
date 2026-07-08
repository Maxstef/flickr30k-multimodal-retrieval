from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from PIL import Image, UnidentifiedImageError
import streamlit as st

from app.config import TOP_K
from app.components.display import (
    get_similarity_color,
    get_similarity_label,
    show_similarity_guide,
)
from app.utils.data_loader import (
    load_caption_index,
    load_mini_clip_model,
    load_vocab,
)
from app.utils.inference import encode_uploaded_image
from app.utils.search import search_top_k


st.set_page_config(
    page_title="Image to Caption",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Image → Caption")

st.write(
    "Upload an image and Mini-CLIP will retrieve the most similar captions "
    "from the prepared Flickr30k caption index."
)

st.subheader("📤 Upload an image")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

top_k = st.slider(
    "Number of captions to retrieve",
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

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded image")
        st.image(image, width="stretch")

    with st.spinner("Searching captions..."):
        vocab = load_vocab()
        model = load_mini_clip_model(vocab_size=len(vocab))
        captions_df, caption_embeddings = load_caption_index()

        image_embedding = encode_uploaded_image(
            image=image,
            model=model,
        )

        results = search_top_k(
            query_embedding=image_embedding,
            candidate_embeddings=caption_embeddings,
            top_k=top_k,
        )

    with col2:
        st.subheader("Retrieved captions")

        for rank, result in enumerate(results, start=1):
            caption = captions_df.iloc[result["index"]]["caption"]
            score = result["score"]
            label = get_similarity_label(score)
            color = get_similarity_color(score)

            with st.container(border=True):
                st.markdown(f"**{rank}. {caption}**")
                st.caption(label)

                st.markdown(
                    f"""
                    Similarity score:
                    <span style="color:{color}; font-weight:600;">
                        {score:.3f}
                    </span>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(max(0.0, min(score, 1.0)))
else:
    st.info("Upload an image to retrieve matching captions.")

st.divider()

with st.expander("About this demo"):
    st.write(
        """
        Captions are retrieved from a prepared Flickr30k caption index. The model
        searches for captions that are semantically similar to the uploaded image,
        so results may describe related concepts rather than the exact image details.
        """
    )

show_similarity_guide()