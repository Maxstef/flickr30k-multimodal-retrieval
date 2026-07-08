import streamlit as st


def render_sidebar():
    st.sidebar.markdown("### 🔎 Mini-CLIP Explorer")
    st.sidebar.caption("Semantic image-text retrieval demo")

    st.sidebar.divider()

    st.sidebar.markdown("**Demo index**")
    st.sidebar.write("1,000 images")
    st.sidebar.write("5,000 captions")

    st.sidebar.markdown("**Model**")
    st.sidebar.write("Mini-CLIP")
    st.sidebar.write("Embedding dim: 256")

    st.sidebar.markdown("**Similarity**")
    st.sidebar.write("Cosine similarity")
