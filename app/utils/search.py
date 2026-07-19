"""Streamlit wrapper around shared retrieval utilities."""

from src.serving.retrieval import retrieve_top_k


def search_top_k(*args, **kwargs):
    """
    Backward-compatible wrapper for the Streamlit application.
    """
    return retrieve_top_k(*args, **kwargs)
