from tokenizer import Tokenizer
import pickle
import sys

with open('data.pkl', 'rb') as file:
    tokenizer: Tokenizer = pickle.load(file)
    tokens = tokenizer.encode("".join(sys.stdin.readlines()))
    print(tokenizer.decode(tokens, colorize=True))
