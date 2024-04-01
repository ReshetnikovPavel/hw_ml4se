import pickle
import sys
from collections import Counter
from itertools import chain

import colorama
import regex as re
from datasets import load_dataset


def most_common(seq: iter):
    counts = Counter(seq)
    return counts.most_common(1)[0][0] if counts else None


def merge(seq: list[int], pair: tuple[int, int], symbol: int) -> list[int]:
    new_seq = []
    i = 0
    while i < len(seq) - 1:
        if (seq[i], seq[i + 1]) == pair:
            new_seq.append(symbol)
            i += 2
        else:
            new_seq.append(seq[i])
            i += 1
    if i == len(seq) - 1:
        new_seq.append(seq[-1])
    return new_seq


class BasicTokenizer:
    def __init__(self):
        self.vocab = {b: bytes([b]) for b in range(256)}
        self.merges = []

    def train(self, text: str, vocab_size: int) -> None:
        sequence = bytes(text, "utf-8")
        new_symbol = 256
        seq = [int(b) for b in sequence]
        merges = dict()

        while len(self.vocab) < vocab_size:
            pairs = list(zip(seq, seq[1:]))
            if not pairs:
                break
            pair = most_common(pairs)
            merges[pair] = new_symbol
            self.vocab[new_symbol] = self.vocab[pair[0]] + self.vocab[pair[1]]

            seq = merge(seq, pair, new_symbol)
            new_symbol += 1

        self.merges = sorted((symbol, pair) for pair, symbol in merges.items())

    def encode(self, text: str) -> list[int]:
        sequence = bytes(text, "utf-8")
        seq = [int(b) for b in sequence]
        for symbol, pair_from_train in self.merges:
            seq = merge(seq, pair_from_train, symbol)
        return seq

    def decode(self, ids: list[int]) -> str:
        return "~".join(self.vocab[id].decode("utf-8") for id in ids)


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def merge_with_cache(
    chunks: list[list[int]], pair: tuple[int, int], symbol: int
) -> None:
    cache = dict()
    for i in range(len(chunks)):
        tuple_chunk = tuple(chunks[i])
        if tuple_chunk in cache:
            chunks[i] = cache[tuple_chunk]
        else:
            res = merge(chunks[i], pair, symbol)
            cache[tuple_chunk] = res
            chunks[i] = res


class Tokenizer:
    def __init__(self, special_tokens: set[str] = set()):
        self.symbol = 256
        self.special_tokens = dict()
        self.vocab = {b: bytes([b]) for b in range(self.symbol)}
        for special_token in special_tokens:
            self.vocab[self.symbol] = bytes(special_token, "utf-8")
            self.special_tokens[special_token] = self.symbol
            self.symbol += 1
        self.merges = []
        self.pattern = self.compile_pattern()

    def compile_pattern(self):
        if not self.special_tokens:
            return re.compile(GPT4_SPLIT_PATTERN)
        else:
            escaped_tokens = [re.escape(t) for t in self.special_tokens]
            return re.compile("|".join(escaped_tokens) + r"|" + GPT4_SPLIT_PATTERN)

    def train(self, text: str, vocab_size: int, progress: bool = False) -> None:
        merges = dict()
        chunks = self.split_into_chunks(text)

        while len(self.vocab) < vocab_size:
            if progress:
                print(f"{len(self.vocab)}/{vocab_size}")

            pair = most_common(
                chain.from_iterable(
                    zip(chunk, chunk[1:]) for chunk in chunks if len(chunk) > 1
                )
            )
            if not pair:
                break

            new_token = self.vocab[pair[0]] + self.vocab[pair[1]]
            if new_token in self.special_tokens:
                symbol = self.special_tokens[new_token]
                merges[pair] = symbol
                merge_with_cache(chunks, pair, symbol)
                continue

            self.vocab[self.symbol] = new_token
            merges[pair] = self.symbol
            merge_with_cache(chunks, pair, self.symbol)
            self.symbol += 1

        self.merges = sorted((symbol, pair) for pair, symbol in merges.items())

    def split_into_chunks(self, text: str) -> list[list[int]]:
        chunks = []
        for chunk in re.findall(self.pattern, text):
            if not chunk:
                continue
            if chunk in self.special_tokens:
                chunks.append([self.special_tokens[chunk]])
            else:
                chunks.append(list(chunk.encode("utf-8")))
        return chunks

    def encode(self, text: str) -> list[int]:
        chunks = self.split_into_chunks(text)
        for symbol, pair_from_train in self.merges:
            merge_with_cache(chunks, pair_from_train, symbol)
        return list(chain.from_iterable(chunks))

    def decode(self, ids: list[int], colorize: bool = False, sep: str = "") -> str:
        if not colorize:
            return sep.join(self.vocab[id].decode("utf-8") for id in ids)
        else:
            colorama.init(autoreset=True)
            decoded = []
            for index, id in enumerate(ids):
                color = colors[index % len(colors)]
                token = self.vocab[id].decode("utf-8")
                decoded.append(color + token)
            return sep.join(decoded)


colors = [
    colorama.Fore.RED,
    colorama.Fore.GREEN,
    colorama.Fore.YELLOW,
    colorama.Fore.BLUE,
    colorama.Fore.MAGENTA,
    colorama.Fore.CYAN,
    colorama.Fore.WHITE,
]

EOF = "<|endoftext|>"

python_common_builtins = [
    "abs",
    "all",
    "any",
    "bin",
    "bool",
    "bytes",
    "callable",
    "chr",
    "dict",
    "dir",
    "enumerate",
    "eval",
    "exec",
    "exit",
    "filter",
    "float",
    "hash",
    "hex",
    "id",
    "input",
    "int",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "open",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "sorted",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "zip",
]

python_keywords = [
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
]

python_operations = [
    "+",
    "-",
    "*",
    "/",
    "%",
    "**",
    "//",
    "&",
    "|",
    "^",
    "~",
    "<<",
    ">>",
    "=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "**=",
    "//=",
    "&=",
    "|=",
    "^=",
    "~=",
    "<<=",
    ">>=",
]


def build_special_symbols() -> set[str]:
    res = set()
    res.add(EOF)
    res.update(python_keywords)
    res.update(python_common_builtins)
    res.update(" " + kw for kw in python_keywords)
    res.update(" " + kw for kw in python_common_builtins)
    res.update(" " + kw for kw in python_operations)
    return res


if __name__ == "__main__":
    dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")["train"]
    text = EOF.join(example["output"] for example in dataset)
    tokenizer = Tokenizer(build_special_symbols())
    tokenizer.train(text, 10000)

    with open("data.pkl", "wb") as file:
        pickle.dump(tokenizer, file)

    print("VOCAB:::")
    for key in tokenizer.vocab:
        try:
            v = tokenizer.vocab[key].decode("utf-8")
        except UnicodeDecodeError:
            v = ""
        print(f'{key}: "{v}"')

    text = str(sys.stdin.read())
    encoded = tokenizer.encode(text)
    print("ENCODED:::")
    print(encoded)
    print()

    decoded = tokenizer.decode(encoded)
    print("DECODED:::")
    print(decoded)
    print()
