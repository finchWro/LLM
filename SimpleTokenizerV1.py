
import re


class SimpleTokenizerV1:
    """Maps string tokens to ints using a fixed vocab; encode splits text, decode joins ids."""

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        # Pattern differs slightly from corpus prep in tokens.py (e.g. -- vs em dash).
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Collapse erroneous space before punctuation after naive join.
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
