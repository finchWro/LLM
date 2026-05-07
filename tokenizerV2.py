import re  # Regular expressions: split text on punctuation and whitespace patterns.
import SimpleTokenizerV2  # Our tokenizer class: maps strings ↔ integer ids using the vocab dict.
import tiktoken

# Load corpus and build a token→id vocabulary from its unique tokens.

with open("The_Verdict.txt", "r", encoding="utf-8") as file:  # Open the story text as UTF-8 so accents etc. read correctly.
    content = file.read()  # Read the entire file into one Python string.

# Capturing group keeps punctuation as separate tokens; \s splits on whitespace.
preprocessed = re.split(r'([,.!?;:\'"()\-—…]|\s)', content)  # Split on spaces/tabs/newlines OR on listed punctuation; parentheses in regex capture those chars as their own pieces.
preprocessed = [item for item in preprocessed if item.strip()]  # Drop empty strings and pure-whitespace chunks left by split.

all_words = sorted(set(preprocessed))  # Unique tokens, sorted alphabetically so vocab order is stable and reproducible.
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)  # How many distinct tokens the corpus vocabulary has (ids will be 0 .. vocab_size-1).
print(f"Vocab size: {vocab_size}")


vocab = {token: integer for integer, token in enumerate(all_words)}  # Map each token string to a consecutive integer id.

tokenizer = SimpleTokenizerV2.SimpleTokenizerV2(vocab)  # Build tokenizer with str→int and reverse int→str lookups.

text1 = "Hello, do you like tea?"  # First example sentence for a mini “batch”.
text2 = "In the sunlit terraces of the palace."  # Second example sentence.

text = " <|endoftext|> ".join((text1, text2))  # Glue sentences with a special separator token (common in GPT-style training data between documents).

print(text)  # Show the combined string so you can see how end-of-text markers sit between segments.

print(tokenizer.decode(tokenizer.encode(text)))

text = (
    "Hello, do you like tea?" + "<|endoftext|> " + "In the sunlit terraces of the palace."
    "of someunknowPlace."
     )
tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)

text = "Akwirw ier"
print(text)
# Print each GPT-2 BPE token on its own line (one-liner). Pieces can look odd — that is normal subword splitting.
print(
    *[
        tokenizer.decode([t])
        for t in tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    ],
    sep="\n",
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)

print(strings)
