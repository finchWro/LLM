import re

text = "Hello, world! This is a test. Let's see how it works."
# Split on whitespace/punctuation; parentheses keep delimiters in the result list.
result = re.split(r'([,.!?;:\'"()\-—…]|\s)', text)
result = [item for item in result if item.strip()]
print(result)
