#!/usr/bin/env python3
"""Toy next-word predictor for a limited language scope.

This script trains a tiny neural language model on a small handcrafted corpus
and demonstrates next-word prediction and generation.

It is intentionally simple and self-contained, using only numpy.
"""

import argparse
import random
import sys
from collections import Counter

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required. Install with: pip install numpy")
    sys.exit(1)


CORPUS = [
    "the cat sat on the mat",
    "the cat chased the mouse",
    "the cat purred and slept",
    "the dog sat on the log",
    "the dog ran in the park",
    "the dog barked at the cat",
    "the bird sang in the tree",
    "the bird flew over the house",
    "the mouse hid under the table",
    "the cat slept under the table",
    "the dog chased the cat",
    "the bird watched the dog",
]

START = "<START>"
END = "<END>"
UNK = "<UNK>"


class TinyLanguageModel:
    def __init__(self, vocab, context_size=3, emb_dim=32, hidden_dim=64, seed=42):
        self.vocab = vocab
        self.idx = {w: i for i, w in enumerate(vocab)}
        self.context_size = context_size
        self.vocab_size = len(vocab)
        rng = np.random.RandomState(seed)
        self.emb = rng.randn(self.vocab_size, emb_dim) * 0.1
        self.w1 = rng.randn(context_size * emb_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.randn(hidden_dim, self.vocab_size) * 0.1
        self.b2 = np.zeros(self.vocab_size)

    def forward(self, context_ids):
        x = self.emb[context_ids].reshape(context_ids.shape[0], -1)
        h = np.tanh(x.dot(self.w1) + self.b1)
        logits = h.dot(self.w2) + self.b2
        return x, h, logits

    @staticmethod
    def softmax(logits):
        z = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def loss_and_gradients(self, context_ids, target_ids):
        x, h, logits = self.forward(context_ids)
        probs = self.softmax(logits)
        n = target_ids.shape[0]
        loss = -np.log(probs[np.arange(n), target_ids] + 1e-12).mean()
        dlogits = probs.copy()
        dlogits[np.arange(n), target_ids] -= 1
        dlogits /= n
        dw2 = h.T.dot(dlogits)
        db2 = dlogits.sum(axis=0)
        dh = dlogits.dot(self.w2.T)
        dx = dh * (1 - h * h)
        dw1 = x.T.dot(dx)
        db1 = dx.sum(axis=0)
        demb = dx.dot(self.w1.T).reshape(context_ids.shape[0], self.context_size, -1)
        return loss, dw2, db2, dw1, db1, demb

    def update(self, grads, lr):
        dw2, db2, dw1, db1, demb, context_ids = grads
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        for i in range(context_ids.shape[0]):
            self.emb[context_ids[i]] -= lr * demb[i]

    def predict_distribution(self, context_tokens):
        ids = [self.idx.get(w, self.idx[UNK]) for w in context_tokens]
        ids = np.array([ids])
        _, _, logits = self.forward(ids)
        return self.softmax(logits)[0]

    def predict_next(self, context_tokens, top_k=5):
        dist = self.predict_distribution(context_tokens)
        best = np.argsort(dist)[::-1][:top_k]
        return [(self.vocab[i], float(dist[i])) for i in best]

    def generate(self, prompt, length=15, temperature=1.0):
        tokens = prompt.split()
        context = [START] * max(0, self.context_size - len(tokens)) + tokens[-self.context_size :]
        generated = []
        for _ in range(length):
            logits = self.predict_distribution(context)
            if temperature != 1.0:
                logits = np.log(logits + 1e-12) / temperature
                logits = np.exp(logits) / np.exp(logits).sum()
            next_id = np.random.choice(len(self.vocab), p=logits)
            next_token = self.vocab[next_id]
            if next_token == END:
                break
            generated.append(next_token)
            context = context[1:] + [next_token]
        return " ".join(generated)


def build_dataset(corpus, context_size):
    tokenized = []
    for sentence in corpus:
        words = sentence.strip().split()
        tokenized.append([START] * context_size + words + [END])
    vocab = [START, END, UNK]
    counter = Counter(w for sent in tokenized for w in sent)
    for word, _ in counter.most_common():
        if word not in vocab:
            vocab.append(word)
    contexts = []
    targets = []
    for sentence in tokenized:
        for i in range(context_size, len(sentence)):
            context = sentence[i - context_size : i]
            target = sentence[i]
            contexts.append(context)
            targets.append(target)
    return vocab, contexts, targets


def encode_examples(contexts, targets, vocab_index):
    context_ids = np.array([ [vocab_index.get(w, vocab_index[UNK]) for w in context] for context in contexts ], dtype=np.int32)
    target_ids = np.array([vocab_index.get(w, vocab_index[UNK]) for w in targets], dtype=np.int32)
    return context_ids, target_ids


def train(model, contexts, targets, epochs=500, lr=0.1, batch_size=16):
    context_ids, target_ids = encode_examples(contexts, targets, model.idx)
    n = context_ids.shape[0]
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            ctx = context_ids[batch]
            tgt = target_ids[batch]
            loss, dw2, db2, dw1, db1, demb = model.loss_and_gradients(ctx, tgt)
            model.update((dw2, db2, dw1, db1, demb, ctx), lr)
            total_loss += loss * len(batch)
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  loss={total_loss / n:.4f}")
    return model


def demo(model):
    print("\nSample next-word predictions:\n")
    samples = [
        [START, START, "the"],
        [START, "the", "cat"],
        ["the", "dog", "sat"],
        ["the", "bird", "flew"],
        ["the", "mouse", "hid"],
    ]
    for context in samples:
        words = [w for w in context if w != START]
        print("context:", " ".join(words) if words else "<start>")
        for word, prob in model.predict_next(context, top_k=5):
            print(f"  {word:10s} {prob:.4f}")
        print()
    print("Generated examples:\n")
    for prompt in ["the cat", "the dog", "the bird", "the mouse"]:
        print(f"prompt: {prompt}")
        print("  ", model.generate(prompt, length=12, temperature=0.8))
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="Toy next-word predictor")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--generate", type=str, default=None, help="Prompt text to generate from")
    parser.add_argument("--length", type=int, default=15, help="Generate length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--demo", action="store_true", help="Run demonstration after training")
    return parser.parse_args()


def main():
    args = parse_args()
    vocab, contexts, targets = build_dataset(CORPUS, context_size=3)
    model = TinyLanguageModel(vocab, context_size=3, emb_dim=32, hidden_dim=64)
    train(model, contexts, targets, epochs=args.epochs, lr=args.lr, batch_size=8)
    if args.demo:
        demo(model)
    if args.generate:
        print("\nGenerated text:")
        print(model.generate(args.generate, length=args.length, temperature=args.temperature))


if __name__ == "__main__":
    main()
