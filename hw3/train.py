import pickle
from datasets import load_dataset

import tokenizer

print("Loading datasets")
dataset = (
    load_dataset("iamtarun/python_code_instructions_18k_alpaca")["train"]
)

n = 30000
path = "data.pkl"

text = tokenizer.EOF.join(example["output"] for example in dataset)
tokenizer = tokenizer.Tokenizer(tokenizer.build_special_symbols())

print("Training started")
tokenizer.train(text, n, progress=True)

print(f"Saving to {path}")
with open(path, "wb") as file:
    pickle.dump(tokenizer, file)

print("VOCAB:::")
for key in tokenizer.vocab:
    try:
        v = tokenizer.vocab[key].decode("utf-8")
    except UnicodeDecodeError:
        v = ""
    print(f'{key}: "{v}"')
