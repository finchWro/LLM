import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
allowed_special = {"<|endoftext|>"}

text = (
    "Hello, do you like tea?" + "<|endoftext|> " + "In the sunlit terraces of the palace."
    "of someunknowPlace."
)
integers = tokenizer.encode(text, allowed_special=allowed_special)
print(integers)
strings = tokenizer.decode(integers)
print(strings)

text = "Akwirw ier"
print(text)
# Print each GPT-2 BPE token on its own line (subwords; odd splits are normal).
print(
    *[
        tokenizer.decode([t])
        for t in tokenizer.encode(text, allowed_special=allowed_special)
    ],
    sep="\n",
)

integers = tokenizer.encode(text, allowed_special=allowed_special)
print(integers)
strings = tokenizer.decode(integers)

print(strings)
