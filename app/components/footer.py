import streamlit as st


def render_footer():
    st.divider()

    st.caption(
        "Developed by Maksym Stefanko • "
        "Mini-CLIP Image–Text Retrieval Demo"
    )

    st.link_button(
        "View source on GitHub",
        "https://github.com/Maxstef/flickr30k-multimodal-retrieval",
    )
