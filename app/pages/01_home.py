import streamlit as st

st.set_page_config(
    page_title="Mini-CLIP Explorer",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Mini-CLIP Explorer")

st.write(
    """
    This application demonstrates a lightweight CLIP-style model trained for
    image-text understanding using contrastive learning.

    The model learns a shared embedding space where matching images and captions
    are placed close together. This enables retrieval and similarity-based
    image-text matching.
    """
)

st.subheader("What you can do")

st.markdown(
    """
    - **Image → Caption**: upload an image and retrieve the most similar captions.
    - **Caption → Image**: enter a text description and find matching images.
    - **Image + Caption Matching**: compare an image and caption using cosine similarity.
    - **Similar Images**: upload an image and find visually or semantically similar images.
    """
)

st.subheader("Model summary")

col1, col2, col3 = st.columns(3)

col1.metric("Validation F1-score", "0.889")
col2.metric("Best threshold", "0.25")
col3.metric("Validation accuracy", "0.888")

st.info(
    """
    The demo uses a prepared subset of Flickr30k images with precomputed
    embeddings to keep the app lightweight and fast.
    """
)

st.subheader("Project Repository")

st.write(
    "Explore the complete project, including notebooks, model implementations, "
    "training pipelines, evaluation experiments, and this Streamlit application."
)

st.link_button(
    "🔗 View Project on GitHub",
    "https://github.com/Maxstef/flickr30k-multimodal-retrieval",
)

st.divider()

st.caption(
    "Developed by Maksym Stefanko • Multimodal Image-Text Retrieval with Mini-CLIP"
)