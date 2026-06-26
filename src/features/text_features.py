from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer(max_features=5000, ngram_range=(1, 1)):
    """
    Create a TF-IDF vectorizer for caption text.

    Args:
        max_features: Maximum number of TF-IDF features.
        ngram_range: Range of n-grams to include.

    Returns:
        Configured TfidfVectorizer.
    """
    return TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        ngram_range=ngram_range,
    )
