from PIL import Image, UnidentifiedImageError
import streamlit as st

from app.config import TOP_K
from app.utils.data_loader import (
    load_caption_index,
    load_mini_clip_model,
    load_vocab,
)
from app.utils.inference import encode_uploaded_image
from app.utils.search import search_top_k
from app.components.display import get_similarity_label, get_similarity_color
from app.components.display import show_similarity_guide

st.set_page_config(
    page_title="Image to Caption",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Image → Caption")

st.write(
    """
    Upload an image and Mini-CLIP will retrieve the most similar captions
    from the prepared Flickr30k caption index.
    """
)

st.info(
    """
    Captions are retrieved from a prepared Flickr30k caption index. The model
    searches for captions that are semantically similar to the uploaded image,
    so results may describe related concepts rather than the exact image details.
    """
)

show_similarity_guide()

top_k = st.slider(
    "Number of captions to retrieve",
    min_value=1,
    max_value=10,
    value=TOP_K,
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, RuntimeError, OSError) as error:
        st.error(
            "Unable to process this image. "
            "Please upload a valid JPG, JPEG, PNG, or WebP image."
        )
        st.caption(f"Details: {error}")
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

            with st.container(border=True):
                st.markdown(f"**{rank}. {caption}**")

                st.caption(label)

                color = get_similarity_color(score)

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

    # with col2:
    #     st.subheader("Retrieved captions")

    #     for rank, result in enumerate(results, start=1):
    #         caption = captions_df.iloc[result["index"]]["caption"]
    #         score = result["score"]

    #         st.markdown(
    #             f"""
    #             **{rank}. {caption}**

    #             Similarity score: `{score:.3f}`
    #             """
    #         )
    #         st.progress(float(max(0.0, min(score, 1.0))))
else:
    st.info("Upload an image to retrieve matching captions.")