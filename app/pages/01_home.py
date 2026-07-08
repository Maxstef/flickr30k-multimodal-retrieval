import streamlit as st
from app.components.footer import render_footer
# from app.components.sidebar import render_sidebar

# render_sidebar()

st.set_page_config(
    page_title="Mini-CLIP Explorer",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Mini-CLIP Explorer")

st.write(
    """
    Explore a lightweight CLIP-style model trained for image-text retrieval
    and similarity-based matching.
    """
)

st.info(
    """
    The app uses a prepared Flickr30k demo index with 1,000 images and 5,000 captions.
    Upload an image, enter a caption, or compare image-text pairs using Mini-CLIP embeddings.
    """
)

st.subheader("Try the app")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        **🖼️ Image → Caption**  
        Upload an image and retrieve the most similar captions.

        **📝 Caption → Image**  
        Enter a text description and find matching images.
        """
    )

with col2:
    st.markdown(
        """
        **🔗 Image + Caption Matching**  
        Compare an image and caption using cosine similarity.

        **🖼️ Similar Images**  
        Upload an image and find visually or semantically similar images.
        """
    )

st.subheader("Model performance")

col1, col2, col3 = st.columns(3)

col1.metric("Validation F1-score", "0.889")
col2.metric("Best threshold", "0.25")
col3.metric("Validation accuracy", "0.888")

with st.expander("About the model"):
    st.write(
        """
        Mini-CLIP learns a shared embedding space where matching images and captions
        are placed close together. This enables both retrieval and binary image-text
        matching using cosine similarity.
        """
    )

st.subheader("Project Repository")

st.write(
    "Explore the full project, including notebooks, model implementations, "
    "training pipelines, evaluation experiments, and this Streamlit application."
)

st.link_button(
    "🔗 View Project on GitHub",
    "https://github.com/Maxstef/flickr30k-multimodal-retrieval",
)

render_footer()
