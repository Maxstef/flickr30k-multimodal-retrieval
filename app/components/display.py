import streamlit as st


def show_similarity_guide():
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
    if score >= 0.67:
        return "#2E8B57"  # green
    if score >= 0.50:
        return "#E6A700"  # amber
    return "#D32F2F"      # red
