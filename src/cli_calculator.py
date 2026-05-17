import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data import EOS, PAD, CalculatorTokenizer
from src.model import CalculatorModel
from src.train import generate_rhs


def tokenizer_path_for(weights_path: str) -> str:
    path = Path(weights_path)
    return str(path.with_name(f"{path.stem}_tokenizer.json"))


def load_model_and_tokenizer(weights_path: str):
    model = CalculatorModel.load_weights(weights_path)
    tok_path = tokenizer_path_for(weights_path)
    tokenizer = CalculatorTokenizer.load(tok_path) if os.path.exists(tok_path) else CalculatorTokenizer()
    return model, tokenizer


def predict_next_tokens(model, tokenizer, text, top_k=8):
    encoded = tokenizer.encode_text(text, add_eos=False)
    if len(encoded) < model.context_size:
        ctx = [tokenizer.pad_id] * (model.context_size - len(encoded)) + list(encoded)
    else:
        ctx = list(encoded[-model.context_size :])
    preds = model.predict_next(np.array(ctx, dtype=np.int32), top_k=top_k)
    return [(tokenizer.idx_to_token[i], p) for i, p in preds]


def main():
    parser = argparse.ArgumentParser(description="Use a trained toy calculator model")
    parser.add_argument("--weights", type=str, default="weight/calculator_transformer.json")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.weights)

    def run_prompt(text: str) -> None:
        if text.endswith("="):
            print(generate_rhs(model, tokenizer, text, max_new_tokens=args.max_new_tokens))
            return
        for token, prob in predict_next_tokens(model, tokenizer, text, top_k=args.top_k):
            if token == PAD:
                continue
            display = "<EOS>" if token == EOS else token
            print(f"{display!r}: {prob:.4f}")

    if args.prompt is not None:
        run_prompt(args.prompt)
        return

    if args.interactive:
        print("Enter a prefix. End with '=' to generate the result. Type 'quit' to exit.")
        while True:
            text = input("> ").strip()
            if text.lower() in {"quit", "exit"}:
                break
            if text:
                run_prompt(text)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
