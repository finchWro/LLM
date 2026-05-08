import tiktoken

with open("The_Verdict.txt", "r", encoding="utf-8") as file:
    content = file.read()



tokenizer = tiktoken.get_encoding("gpt2")

enc_text = tokenizer.encode(content)

print(len(enc_text))

enc_sample = enc_text[50:]

context_size =  4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x:  {x}")
print(f"y:      {y}")

for i in range(1,context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context,"---->",desired)
    print(tokenizer.decode(context),"---->",tokenizer.decode([desired]))





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
)

integers = tokenizer.encode(text, allowed_special=allowed_special)
print(integers)
strings = tokenizer.decode(integers)

print(strings)
