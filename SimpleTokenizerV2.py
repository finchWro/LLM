import re  # Used below to split input into tokens and to fix spacing around punctuation when decoding.


class SimpleTokenizerV2:  # Version 2: unknown words map to a placeholder token instead of crashing.
    def __init__(self, vocab):  # vocab is a dict: token string → integer id (built from the corpus).
        self.str_to_int = dict(vocab)  # Copy so we can register <|unk|> without mutating the caller's dict.
        if "<|unk|>" not in self.str_to_int:
            next_id = max(self.str_to_int.values(), default=-1) + 1
            self.str_to_int["<|unk|>"] = next_id
        self.int_to_str = {i: s for s, i in self.str_to_int.items()}  # Reverse map: id → string for decoding.

    def encode(self, text):  # Turn a human-readable string into a list of integer token ids.
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)  # Split on whitespace or punctuation/dashes; capture group keeps delimiters as separate tokens.
        preprocessed = [item.strip() for item in preprocessed if item.strip()]  # Trim each piece and remove empties.
        unk_id = self.str_to_int["<|unk|>"]  # Guaranteed by __init__.
        ids = [self.str_to_int.get(item, unk_id) for item in preprocessed]  # OOV tokens use the shared unk id.
        return ids  # Return the sequence of integers the model would consume.

    def decode(self, ids):  # Turn a list of ids back into readable text (approximate inverse of encode).
        text = " ".join([self.int_to_str[i] for i in ids])  # Map each id to its string and join with single spaces.
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)  # Remove the artificial space before punctuation so "Hello ," becomes "Hello," again.
        return text  # Final reconstructed string.
