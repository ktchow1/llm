import json
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


DIGITS = "0123456789"
OPERATORS = "+-*/"
DEFAULT_TRAIN_OPERATORS = "+-*"
EQUATION = "="

PAD = "<PAD>"
EOS = "<EOS>"

# Backward-compatible names used by older scripts in this folder.
START = PAD
END = EOS
UNK = PAD


def floor_divide(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("division by zero is not part of this dataset")
    return a // b


def evaluate_binary(a: int, op: str, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return floor_divide(a, b)
    raise ValueError(f"unknown operator: {op}")


def format_sample(a: int, op: str, b: int) -> str:
    return f"{a}{op}{b}={evaluate_binary(a, op, b)}"


def generate_corpus(
    num_expr: int = 5000,
    min_value: int = 0,
    max_value: int = 20,
    operators: Sequence[str] = tuple(DEFAULT_TRAIN_OPERATORS),
    seed: int | None = None,
) -> List[str]:
    rng = random.Random(seed)
    valid_operators = tuple(operators)
    for op in valid_operators:
        if op not in OPERATORS:
            raise ValueError(f"unsupported operator: {op}")

    corpus = [
        format_sample(a, op, b)
        for op in valid_operators
        for a in range(min_value, max_value + 1)
        for b in range(min_value, max_value + 1)
        if not (op == "/" and b == 0)
    ]
    rng.shuffle(corpus)

    if len(corpus) >= num_expr:
        return corpus[:num_expr]

    while len(corpus) < num_expr:
        a = rng.randint(min_value, max_value)
        b = rng.randint(min_value, max_value)
        op = rng.choice(valid_operators)
        if op == "/" and b == 0:
            continue
        corpus.append(format_sample(a, op, b))

    return corpus


@dataclass
class CalculatorTokenizer:
    token_list: List[str] | None = None

    def __post_init__(self) -> None:
        if self.token_list is None:
            self.token_list = [PAD, EOS] + list(DIGITS + OPERATORS + EQUATION)
        self.token_to_idx = {token: i for i, token in enumerate(self.token_list)}
        self.idx_to_token = {i: token for token, i in self.token_to_idx.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_idx[PAD]

    @property
    def eos_id(self) -> int:
        return self.token_to_idx[EOS]

    def tokenize(self, text: str, add_eos: bool = False) -> List[str]:
        tokens: List[str] = []
        for ch in text:
            if ch.isspace():
                continue
            if ch not in self.token_to_idx or ch in (PAD, EOS):
                raise ValueError(f"character {ch!r} is not in the calculator vocabulary")
            tokens.append(ch)
        if add_eos:
            tokens.append(EOS)
        return tokens

    def encode(self, tokens: Iterable[str]) -> np.ndarray:
        return np.array([self.token_to_idx[token] for token in tokens], dtype=np.int32)

    def encode_text(self, text: str, add_eos: bool = False) -> np.ndarray:
        return self.encode(self.tokenize(text, add_eos=add_eos))

    def decode(self, indices: Iterable[int], stop_at_eos: bool = False) -> List[str]:
        tokens: List[str] = []
        for index in indices:
            token = self.idx_to_token[int(index)]
            if stop_at_eos and token == EOS:
                break
            tokens.append(token)
        return tokens

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"tokens": self.token_list}, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CalculatorTokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(token_list=data["tokens"])


def build_training_data(
    corpus: Sequence[str],
    tokenizer: CalculatorTokenizer,
    context_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    contexts: List[np.ndarray] = []
    targets: List[int] = []

    for sample in corpus:
        encoded = tokenizer.encode_text(sample, add_eos=True)
        for i, target in enumerate(encoded):
            prefix = encoded[max(0, i - context_size) : i]
            if len(prefix) < context_size:
                pad = np.full(context_size - len(prefix), tokenizer.pad_id, dtype=np.int32)
                context = np.concatenate([pad, prefix])
            else:
                context = prefix
            contexts.append(context.astype(np.int32))
            targets.append(int(target))

    if not contexts:
        return (
            np.empty((0, context_size), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )
    return np.array(contexts, dtype=np.int32), np.array(targets, dtype=np.int32)


def split_corpus(
    corpus: Sequence[str],
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    if validation_fraction <= 0:
        return list(corpus), []
    shuffled = list(corpus)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * validation_fraction))
    return shuffled[val_count:], shuffled[:val_count]
