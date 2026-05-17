import json
from typing import List, Tuple
import numpy as np


def stable_softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def softmax_grad(soft: np.ndarray, grad: np.ndarray) -> np.ndarray:
    s_grad = grad * soft
    return s_grad - soft * np.sum(s_grad, axis=-1, keepdims=True)


class CalculatorModel:
    def __init__(
        self,
        vocab_size: int,
        context_size: int = 12,
        emb_dim: int = 128,
        ff_dim: int = 512,
        num_layers: int = 1,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.emb_dim = emb_dim
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        rng = np.random.RandomState(seed)

        self.emb = rng.randn(vocab_size, emb_dim).astype(np.float32) * 0.1
        self.pos_emb = rng.randn(context_size, emb_dim).astype(np.float32) * 0.1

        self.Wq = rng.randn(emb_dim, emb_dim).astype(np.float32) * 0.1
        self.Wk = rng.randn(emb_dim, emb_dim).astype(np.float32) * 0.1
        self.Wv = rng.randn(emb_dim, emb_dim).astype(np.float32) * 0.1
        self.Wo = rng.randn(emb_dim, emb_dim).astype(np.float32) * 0.1

        self.W1 = rng.randn(emb_dim, ff_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(ff_dim, dtype=np.float32)
        self.W2 = rng.randn(ff_dim, emb_dim).astype(np.float32) * 0.1
        self.b2 = np.zeros(emb_dim, dtype=np.float32)

        self.Wout = rng.randn(emb_dim, vocab_size).astype(np.float32) * 0.1
        self.bout = np.zeros(vocab_size, dtype=np.float32)

    def forward(self, context_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        batch_size, seq_len = context_ids.shape
        x = self.emb[context_ids] + self.pos_emb[:seq_len]

        caches = []
        # apply the same transformer block `num_layers` times (weight-sharing)
        for _ in range(self.num_layers):
            q = x.dot(self.Wq)
            k = x.dot(self.Wk)
            v = x.dot(self.Wv)

            scale = np.sqrt(self.emb_dim).astype(np.float32)
            scores = q @ k.transpose(0, 2, 1) / scale
            mask = np.triu(np.ones((seq_len, seq_len), dtype=np.bool_), 1)
            scores = np.where(mask[None, :, :], -1e9, scores)

            attn = stable_softmax(scores)
            attn_out = attn @ v
            x2 = x + attn_out.dot(self.Wo)

            ff_pre = x2.dot(self.W1) + self.b1
            ff = np.tanh(ff_pre)
            x3 = x2 + ff.dot(self.W2)

            caches.append(
                {
                    "x": x,
                    "x2": x2,
                    "x3": x3,
                    "q": q,
                    "k": k,
                    "v": v,
                    "scores": scores,
                    "attn": attn,
                    "attn_out": attn_out,
                    "ff": ff,
                    "ff_pre": ff_pre,
                }
            )
            x = x3

        out = x[:, -1, :]
        logits = out.dot(self.Wout) + self.bout

        cache = {"caches": caches, "context_ids": context_ids, "out": out}
        return out, logits, cache

    def loss_and_gradients(self, context_ids: np.ndarray, target_ids: np.ndarray):
        out, logits, cache = self.forward(context_ids)
        probs = stable_softmax(logits)
        n = target_ids.shape[0]

        loss = -np.log(probs[np.arange(n), target_ids] + 1e-12).mean()

        dlogits = probs
        dlogits[np.arange(n), target_ids] -= 1
        dlogits /= n

        dWout = cache["out"].T.dot(dlogits)
        dbout = dlogits.sum(axis=0)
        dout = dlogits.dot(self.Wout.T)

        # Initialize gradient through stacked layers: only last token receives gradient from logits
        last_cache = cache["caches"][-1]
        dx = np.zeros_like(last_cache["x3"])
        dx[:, -1, :] = dout

        # Accumulate gradients for shared weights over each layer (backprop through repeated blocks)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dWo = np.zeros_like(self.Wo)
        dWv = np.zeros_like(self.Wv)
        dWk = np.zeros_like(self.Wk)
        dWq = np.zeros_like(self.Wq)

        # Backprop through each repeated layer (in reverse order)
        for layer_cache in reversed(cache["caches"]):
            # dx here corresponds to gradient w.r.t x3 of this layer
            dx3 = dx

            # FF gradients
            dW2 += layer_cache["ff"].reshape(-1, self.ff_dim).T.dot(
                dx3.reshape(-1, self.emb_dim)
            )
            db2 += dx3.reshape(-1, self.emb_dim).sum(axis=0)

            dff = dx3.dot(self.W2.T)
            dff_pre = dff * (1 - layer_cache["ff"] ** 2)

            dW1 += layer_cache["x2"].reshape(-1, self.emb_dim).T.dot(
                dff_pre.reshape(-1, self.ff_dim)
            )
            db1 += dff_pre.reshape(-1, self.ff_dim).sum(axis=0)

            dx2 = dx3 + dff_pre.dot(self.W1.T)

            # Attention output projection
            dWo += layer_cache["attn_out"].reshape(-1, self.emb_dim).T.dot(
                dx2.reshape(-1, self.emb_dim)
            )
            dattn_out = dx2.dot(self.Wo.T)

            d_attn = dattn_out @ layer_cache["v"].transpose(0, 2, 1)
            dv = layer_cache["attn"].transpose(0, 2, 1) @ dattn_out

            d_scores = softmax_grad(layer_cache["attn"], d_attn)
            scale = np.sqrt(self.emb_dim).astype(np.float32)
            dq = d_scores @ layer_cache["k"] / scale
            dk = d_scores.transpose(0, 2, 1) @ layer_cache["q"] / scale

            dWq += layer_cache["x"].reshape(-1, self.emb_dim).T.dot(
                dq.reshape(-1, self.emb_dim)
            )
            dWk += layer_cache["x"].reshape(-1, self.emb_dim).T.dot(
                dk.reshape(-1, self.emb_dim)
            )
            dWv += layer_cache["x"].reshape(-1, self.emb_dim).T.dot(
                dv.reshape(-1, self.emb_dim)
            )

            # gradient w.r.t. input x of this layer
            dx = (
                dq.dot(self.Wq.T)
                + dk.dot(self.Wk.T)
                + dv.dot(self.Wv.T)
                + dx2
            )

        # After backprop through all layers, dx is gradient wrt initial embeddings+pos
        dpos = dx.sum(axis=0)
        demb = np.zeros_like(self.emb)
        for b in range(cache["context_ids"].shape[0]):
            for t in range(cache["context_ids"].shape[1]):
                demb[cache["context_ids"][b, t]] += dx[b, t]

        return (
            loss,
            dWout,
            dbout,
            dW2,
            db2,
            dW1,
            db1,
            dWo,
            dWv,
            dWk,
            dWq,
            demb,
            dpos,
        )

    def update(self, grads, lr: float, grad_clip: float | None = 1.0):
        (
            dWout,
            dbout,
            dW2,
            db2,
            dW1,
            db1,
            dWo,
            dWv,
            dWk,
            dWq,
            demb,
            dpos,
        ) = grads
        if grad_clip is not None:
            grads_to_clip = [
                dWout,
                dbout,
                dW2,
                db2,
                dW1,
                db1,
                dWo,
                dWv,
                dWk,
                dWq,
                demb,
                dpos,
            ]
            total_norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads_to_clip))
            if total_norm > grad_clip:
                scale = grad_clip / (total_norm + 1e-12)
                (
                    dWout,
                    dbout,
                    dW2,
                    db2,
                    dW1,
                    db1,
                    dWo,
                    dWv,
                    dWk,
                    dWq,
                    demb,
                    dpos,
                ) = [g * scale for g in grads_to_clip]

        self.Wout -= lr * dWout
        self.bout -= lr * dbout
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.Wo -= lr * dWo
        self.Wv -= lr * dWv
        self.Wk -= lr * dWk
        self.Wq -= lr * dWq
        self.pos_emb -= lr * dpos
        # demb is accumulated per-vocabulary row; apply directly to embeddings
        self.emb -= lr * demb

    def predict_distribution(self, context_ids: np.ndarray) -> np.ndarray:
        context_ids = np.array([context_ids], dtype=np.int32)
        _, logits, _ = self.forward(context_ids)
        return stable_softmax(logits)[0]

    def predict_next(self, context_ids: np.ndarray, top_k: int = 10):
        dist = self.predict_distribution(context_ids)
        best_idx = np.argsort(dist)[::-1][:top_k]
        return [(i, float(dist[i])) for i in best_idx]

    def generate_greedy(self, start_tokens: List[int], max_length: int = 32):
        context = list(start_tokens)
        out = []
        for _ in range(max_length):
            if len(context) < self.context_size:
                c = [0] * (self.context_size - len(context)) + context
            else:
                c = context[-self.context_size:]
            dist = self.predict_distribution(np.array(c, dtype=np.int32))
            next_idx = int(np.argmax(dist))
            if next_idx in (0, 1):
                break
            out.append(next_idx)
            context.append(next_idx)
        return out

    def save_weights(self, filepath: str):
        weights = {
            "vocab_size": int(self.vocab_size),
            "context_size": int(self.context_size),
            "emb_dim": int(self.emb_dim),
            "ff_dim": int(self.ff_dim),
            "num_layers": int(self.num_layers),
            "emb": self.emb.tolist(),
            "pos_emb": self.pos_emb.tolist(),
            "Wq": self.Wq.tolist(),
            "Wk": self.Wk.tolist(),
            "Wv": self.Wv.tolist(),
            "Wo": self.Wo.tolist(),
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "Wout": self.Wout.tolist(),
            "bout": self.bout.tolist(),
        }
        with open(filepath, "w") as f:
            json.dump(weights, f)

    @staticmethod
    def load_weights(filepath: str):
        with open(filepath, "r") as f:
            weights = json.load(f)
        model = CalculatorModel(
            vocab_size=weights["vocab_size"],
            context_size=weights["context_size"],
            emb_dim=weights["emb_dim"],
            ff_dim=weights["ff_dim"],
            num_layers=weights.get("num_layers", 1),
        )
        model.emb = np.array(weights["emb"], dtype=np.float32)
        model.pos_emb = np.array(weights["pos_emb"], dtype=np.float32)
        model.Wq = np.array(weights["Wq"], dtype=np.float32)
        model.Wk = np.array(weights["Wk"], dtype=np.float32)
        model.Wv = np.array(weights["Wv"], dtype=np.float32)
        model.Wo = np.array(weights["Wo"], dtype=np.float32)
        model.W1 = np.array(weights["W1"], dtype=np.float32)
        model.b1 = np.array(weights["b1"], dtype=np.float32)
        model.W2 = np.array(weights["W2"], dtype=np.float32)
        model.b2 = np.array(weights["b2"], dtype=np.float32)
        model.Wout = np.array(weights["Wout"], dtype=np.float32)
        model.bout = np.array(weights["bout"], dtype=np.float32)
        return model
