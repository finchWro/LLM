import re
import SimpleTokenizerV1

# Load corpus and build a token→id vocabulary from its unique tokens.

with open("The_Verdict.txt", "r", encoding="utf-8") as file:
    content = file.read()

# Capturing group keeps punctuation as separate tokens; \s splits on whitespace.
preprocessed = re.split(r'([,.!?;:\'"()\-—…]|\s)', content)
preprocessed = [item for item in preprocessed if item.strip()]

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

vocab = {token: integer for integer, token in enumerate(all_words)}

tokenizer = SimpleTokenizerV1.SimpleTokenizerV1(vocab)

text = "see"
encoded = tokenizer.encode(text)
print("Encoded:", encoded)
