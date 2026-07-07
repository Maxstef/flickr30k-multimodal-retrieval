from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
from PIL import Image

from app.config import APP_IMAGES_DIR, TOP_K
from app.components.display import (
    show_similarity_guide,
    get_similarity_label,
    get_similarity_color,
    make_thumbnail,
)
from app.utils.data_loader import (
    load_caption_index,
    load_image_index,
    load_mini_clip_model,
    load_vocab,
)
from app.utils.inference import encode_text_query
from app.utils.search import search_top_k


st.set_page_config(
    page_title="Caption to Image",
    page_icon="📝",
    layout="wide",
)

st.title("📝 Caption → Image")

st.write(
    """
    Enter a text description and Mini-CLIP will retrieve the most similar images
    from the prepared Flickr30k image index.
    """
)

st.info(
    """
    Images are retrieved from a prepared demo index of Flickr30k images. Results
    are based on semantic similarity, so retrieved images may match the general
    meaning of the caption rather than every exact detail.
    """
)

show_similarity_guide()

top_k = st.slider(
    "Number of images to retrieve",
    min_value=1,
    max_value=10,
    value=TOP_K,
)

query = st.text_input(
    "Enter a caption or description",
    placeholder="A dog running through a grassy field",
)

if query:
    with st.spinner("Searching images..."):
        vocab = load_vocab()
        model = load_mini_clip_model(vocab_size=len(vocab))
        images_df, image_embeddings = load_image_index()
        captions_df, _ = load_caption_index()

        query_embedding = encode_text_query(
            text=query,
            model=model,
            vocab=vocab,
        )

        results = search_top_k(
            query_embedding=query_embedding,
            candidate_embeddings=image_embeddings,
            top_k=top_k,
        )

    st.subheader("Top matching images")

    cols = st.columns(min(top_k, 3))

    for rank, result in enumerate(results, start=1):
        image_row = images_df.iloc[result["index"]]
        image_path = APP_IMAGES_DIR / image_row["filename"]

        score = result["score"]
        label = get_similarity_label(score)
        color = get_similarity_color(score)

        with cols[(rank - 1) % len(cols)]:
            with st.container(border=True):
                image = Image.open(image_path)
                thumbnail = make_thumbnail(image)
                st.image(thumbnail, width="stretch")

                st.markdown(f"**{rank}. {label}**")

                st.markdown(
                    f"""
                    Similarity score:
                    <span style="color:{color}; font-weight:600;">
                        {score:.3f}
                    </span>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(float(max(0.0, min(score, 1.0))))

                image_idx = image_row["image_idx"]

                image_captions = captions_df[
                    captions_df["image_idx"] == image_idx
                ]["caption"].tolist()

                with st.expander("Original Flickr30k captions"):
                    for caption in image_captions:
                        st.markdown(f"- {caption}")
else:
    st.info("Enter a caption to retrieve matching images.")
