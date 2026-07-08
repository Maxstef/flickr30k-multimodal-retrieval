import re
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "with", "of", "to",
    "is", "are", "was", "were", "be", "being", "by", "for", "from", "as",
    "this", "that", "these", "those", "it", "its", "his", "her", "their",
    "man", "woman", "person", "people",
}


def tokenize_caption(text):
    """
    Extract simple lowercase word tokens from a caption.
    """
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def extract_caption_keywords(captions, top_n=8):
    """
    Extract frequent non-stopword keywords from a list of captions.
    """
    counter = Counter()

    for caption in captions:
        tokens = tokenize_caption(caption)
        tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 2]
        counter.update(tokens)

    return [token for token, _ in counter.most_common(top_n)]


def explain_image_similarity(query_captions, retrieved_captions, top_n=5):
    """
    Create a short explanation based on overlapping caption keywords.
    """
    query_keywords = set(extract_caption_keywords(query_captions, top_n=12))
    retrieved_keywords = set(extract_caption_keywords(retrieved_captions, top_n=12))

    shared_keywords = sorted(query_keywords & retrieved_keywords)

    if shared_keywords:
        keywords = ", ".join(shared_keywords[:top_n])
        return f"Likely similar because both images are associated with: **{keywords}**."

    retrieved_summary = ", ".join(list(retrieved_keywords)[:top_n])

    if retrieved_summary:
        return (
            "The retrieved image does not share many exact caption keywords, "
            f"but it is associated with related visual concepts such as: **{retrieved_summary}**."
        )

    return "No clear caption-based explanation was found."