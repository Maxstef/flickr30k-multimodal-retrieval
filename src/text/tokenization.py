import re
from collections import Counter

from src.config import (
    PAD_TOKEN,
    UNK_TOKEN,
    PAD_IDX,
    UNK_IDX,
)


def simple_tokenize(text):
    """
    Tokenize text using lowercase conversion and basic word extraction.
    """
    return re.findall(r"\b\w+\b", text.lower())


def build_vocab(captions, min_freq=1):
    """
    Build a word-to-index vocabulary from training captions.

    Args:
        captions: Iterable of caption strings.
        min_freq: Minimum frequency required for a token to be included.

    Returns:
        Dictionary mapping tokens to integer IDs.
    """
    counter = Counter()

    for caption in captions:
        counter.update(simple_tokenize(caption))

    vocab = {
        PAD_TOKEN: PAD_IDX,
        UNK_TOKEN: UNK_IDX,
    }

    for token, count in counter.items():
        if count >= min_freq:
            vocab[token] = len(vocab)

    return vocab


def encode_caption(caption, vocab, max_length=32):
    """
    Convert a caption into a fixed-length list of token IDs.
    """
    tokens = simple_tokenize(caption)

    token_ids = [
        vocab.get(token, UNK_IDX)
        for token in tokens
    ]

    token_ids = token_ids[:max_length]

    padding_length = max_length - len(token_ids)
    token_ids += [PAD_IDX] * padding_length

    return token_ids