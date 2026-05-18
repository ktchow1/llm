import json
from typing import Dict, List, Tuple

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
        context_size: int = 32,
        emb_dim: int = 64,
        ff_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        seed: int = 42,
    ):
        if emb_dim % num_heads != 0:
            raise ValueError("--emb-dim must be divisible by --num-heads")
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.emb_dim = emb_dim
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.adam_step = 0
        self.adam_m: Dict[str, np.ndarray] = {}
        self.adam_v: Dict[str, np.ndarray] = {}

        rng = np.random.RandomState(seed)
        self.emb = rng.randn(vocab_size, emb_dim).astype(np.float32) * 0.02
        self.pos_emb = rng.randn(context_size, emb_dim).astype(np.float32) * 0.02
        self.layers = [self._new_layer(rng) for _ in range(num_layers)]
        self.Wout = rng.randn(emb_dim, vocab_size).astype(np.float32) * 0.02
        self.bout = np.zeros(vocab_size, dtype=np.float32)

    def _new_layer(self, rng: np.random.RandomState) -> Dict[str, np.ndarray]:
        scale = 0.02
        return {
            "ln1_g": np.ones(self.emb_dim, dtype=np.float32),
            "ln1_b": np.zeros(self.emb_dim, dtype=np.float32),
            "Wq": rng.randn(self.emb_dim, self.emb_dim).astype(np.float32) * scale,
            "Wk": rng.randn(self.emb_dim, self.emb_dim).astype(np.float32) * scale,
            "Wv": rng.randn(self.emb_dim, self.emb_dim).astype(np.float32) * scale,
            "Wo": rng.randn(self.emb_dim, self.emb_dim).astype(np.float32) * scale,
            "ln2_g": np.ones(self.emb_dim, dtype=np.float32),
            "ln2_b": np.zeros(self.emb_dim, dtype=np.float32),
            "W1": rng.randn(self.emb_dim, self.ff_dim).astype(np.float32) * scale,
            "b1": np.zeros(self.ff_dim, dtype=np.float32),
            "W2": rng.randn(self.ff_dim, self.emb_dim).astype(np.float32) * scale,
            "b2": np.zeros(self.emb_dim, dtype=np.float32),
        }

    def forward(self, context_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        _, seq_len = context_ids.shape
        x = self.emb[context_ids] + self.pos_emb[:seq_len]

        caches = []
        for layer in self.layers:
            ln1, ln1_cache = self._layer_norm(x, layer["ln1_g"], layer["ln1_b"])
            q = ln1.dot(layer["Wq"])
            k = ln1.dot(layer["Wk"])
            v = ln1.dot(layer["Wv"])
            qh = self._split_heads(q)
            kh = self._split_heads(k)
            vh = self._split_heads(v)

            scale = np.sqrt(self.head_dim).astype(np.float32)
            scores = qh @ kh.transpose(0, 1, 3, 2) / scale
            mask = np.triu(np.ones((seq_len, seq_len), dtype=np.bool_), 1)
            scores = np.where(mask[None, None, :, :], -1e9, scores)

            attn = stable_softmax(scores)
            attn_heads = attn @ vh
            attn_out = self._merge_heads(attn_heads)
            x2 = x + attn_out.dot(layer["Wo"])

            ln2, ln2_cache = self._layer_norm(x2, layer["ln2_g"], layer["ln2_b"])
            ff_pre = ln2.dot(layer["W1"]) + layer["b1"]
            ff = np.tanh(ff_pre)
            x3 = x2 + ff.dot(layer["W2"]) + layer["b2"]

            caches.append(
                {
                    "x": x,
                    "ln1": ln1,
                    "ln1_cache": ln1_cache,
                    "x2": x2,
                    "ln2": ln2,
                    "ln2_cache": ln2_cache,
                    "x3": x3,
                    "q": q,
                    "k": k,
                    "v": v,
                    "qh": qh,
                    "kh": kh,
                    "vh": vh,
                    "attn": attn,
                    "attn_heads": attn_heads,
                    "attn_out": attn_out,
                    "ff": ff,
                }
            )
            x = x3

        out = x[:, -1, :]
        logits = out.dot(self.Wout) + self.bout
        return out, logits, {"caches": caches, "context_ids": context_ids, "out": out}

    def loss_and_gradients(self, context_ids: np.ndarray, target_ids: np.ndarray):
        out, logits, cache = self.forward(context_ids)
        probs = stable_softmax(logits)
        n = target_ids.shape[0]
        loss = -np.log(probs[np.arange(n), target_ids] + 1e-12).mean()

        dlogits = probs.copy()
        dlogits[np.arange(n), target_ids] -= 1
        dlogits /= n

        grads = {
            "Wout": cache["out"].T.dot(dlogits),
            "bout": dlogits.sum(axis=0),
            "layers": [self._zero_layer_grads() for _ in self.layers],
            "emb": np.zeros_like(self.emb),
            "pos_emb": None,
        }
        dout = dlogits.dot(self.Wout.T)

        dx = np.zeros_like(cache["caches"][-1]["x3"])
        dx[:, -1, :] = dout

        for layer_index in reversed(range(len(self.layers))):
            layer = self.layers[layer_index]
            layer_cache = cache["caches"][layer_index]
            layer_grads = grads["layers"][layer_index]
            dx3 = dx

            layer_grads["W2"] = layer_cache["ff"].reshape(-1, self.ff_dim).T.dot(
                dx3.reshape(-1, self.emb_dim)
            )
            layer_grads["b2"] = dx3.reshape(-1, self.emb_dim).sum(axis=0)

            dff = dx3.dot(layer["W2"].T)
            dff_pre = dff * (1 - layer_cache["ff"] ** 2)

            layer_grads["W1"] = layer_cache["ln2"].reshape(-1, self.emb_dim).T.dot(
                dff_pre.reshape(-1, self.ff_dim)
            )
            layer_grads["b1"] = dff_pre.reshape(-1, self.ff_dim).sum(axis=0)

            dln2 = dff_pre.dot(layer["W1"].T)
            dx2_from_ln, layer_grads["ln2_g"], layer_grads["ln2_b"] = self._layer_norm_backward(
                dln2, layer_cache["ln2_cache"]
            )
            dx2 = dx3 + dx2_from_ln

            layer_grads["Wo"] = layer_cache["attn_out"].reshape(-1, self.emb_dim).T.dot(
                dx2.reshape(-1, self.emb_dim)
            )
            dattn_out = dx2.dot(layer["Wo"].T)

            dattn_heads = self._split_heads(dattn_out)
            d_attn = dattn_heads @ layer_cache["vh"].transpose(0, 1, 3, 2)
            dvh = layer_cache["attn"].transpose(0, 1, 3, 2) @ dattn_heads

            d_scores = softmax_grad(layer_cache["attn"], d_attn)
            scale = np.sqrt(self.head_dim).astype(np.float32)
            dqh = d_scores @ layer_cache["kh"] / scale
            dkh = d_scores.transpose(0, 1, 3, 2) @ layer_cache["qh"] / scale
            dq = self._merge_heads(dqh)
            dk = self._merge_heads(dkh)
            dv = self._merge_heads(dvh)

            layer_grads["Wq"] = layer_cache["ln1"].reshape(-1, self.emb_dim).T.dot(
                dq.reshape(-1, self.emb_dim)
            )
            layer_grads["Wk"] = layer_cache["ln1"].reshape(-1, self.emb_dim).T.dot(
                dk.reshape(-1, self.emb_dim)
            )
            layer_grads["Wv"] = layer_cache["ln1"].reshape(-1, self.emb_dim).T.dot(
                dv.reshape(-1, self.emb_dim)
            )

            dln1 = dq.dot(layer["Wq"].T) + dk.dot(layer["Wk"].T) + dv.dot(layer["Wv"].T)
            dx_from_ln1, layer_grads["ln1_g"], layer_grads["ln1_b"] = self._layer_norm_backward(
                dln1, layer_cache["ln1_cache"]
            )
            dx = dx2 + dx_from_ln1

        grads["pos_emb"] = dx.sum(axis=0)
        for b in range(cache["context_ids"].shape[0]):
            for t in range(cache["context_ids"].shape[1]):
                grads["emb"][cache["context_ids"][b, t]] += dx[b, t]

        return loss, grads

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        batch, _, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.emb_dim)

    def _layer_norm(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        eps: float = 1e-5,
    ) -> Tuple[np.ndarray, dict]:
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        inv_std = 1.0 / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + eps)
        x_hat = centered * inv_std
        y = x_hat * gamma + beta
        return y, {"x_hat": x_hat, "inv_std": inv_std, "gamma": gamma}

    def _layer_norm_backward(self, dy: np.ndarray, cache: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_hat = cache["x_hat"]
        inv_std = cache["inv_std"]
        gamma = cache["gamma"]
        dim = dy.shape[-1]
        dgamma = np.sum(dy * x_hat, axis=(0, 1))
        dbeta = np.sum(dy, axis=(0, 1))
        dx_hat = dy * gamma
        dx = (
            inv_std
            / dim
            * (
                dim * dx_hat
                - np.sum(dx_hat, axis=-1, keepdims=True)
                - x_hat * np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
            )
        )
        return dx, dgamma.astype(np.float32), dbeta.astype(np.float32)

    def _zero_layer_grads(self) -> Dict[str, np.ndarray]:
        return {
            "ln1_g": np.zeros(self.emb_dim, dtype=np.float32),
            "ln1_b": np.zeros(self.emb_dim, dtype=np.float32),
            "Wq": np.zeros((self.emb_dim, self.emb_dim), dtype=np.float32),
            "Wk": np.zeros((self.emb_dim, self.emb_dim), dtype=np.float32),
            "Wv": np.zeros((self.emb_dim, self.emb_dim), dtype=np.float32),
            "Wo": np.zeros((self.emb_dim, self.emb_dim), dtype=np.float32),
            "ln2_g": np.zeros(self.emb_dim, dtype=np.float32),
            "ln2_b": np.zeros(self.emb_dim, dtype=np.float32),
            "W1": np.zeros((self.emb_dim, self.ff_dim), dtype=np.float32),
            "b1": np.zeros(self.ff_dim, dtype=np.float32),
            "W2": np.zeros((self.ff_dim, self.emb_dim), dtype=np.float32),
            "b2": np.zeros(self.emb_dim, dtype=np.float32),
        }

    def update(
        self,
        grads,
        lr: float,
        grad_clip: float | None = 1.0,
        weight_decay: float = 0.01,
    ):
        if isinstance(grads, tuple):
            if len(grads) != 1:
                raise ValueError("new CalculatorModel.update expects one gradients dict")
            grads = grads[0]
        if grad_clip is not None:
            self._clip_gradients(grads, grad_clip)

        self.adam_step += 1
        self._adamw_update("emb", self.emb, grads["emb"], lr, weight_decay)
        self._adamw_update("pos_emb", self.pos_emb, grads["pos_emb"], lr, weight_decay)
        self._adamw_update("Wout", self.Wout, grads["Wout"], lr, weight_decay)
        self._adamw_update("bout", self.bout, grads["bout"], lr, 0.0)
        for i, layer in enumerate(self.layers):
            for name, param in layer.items():
                decay = 0.0 if name.startswith("b") or name.endswith("_b") or name.endswith("_g") else weight_decay
                self._adamw_update(
                    f"layers.{i}.{name}",
                    param,
                    grads["layers"][i][name],
                    lr,
                    decay,
                )

    def _clip_gradients(self, grads, grad_clip: float) -> None:
        arrays = [grads["emb"], grads["pos_emb"], grads["Wout"], grads["bout"]]
        for layer_grads in grads["layers"]:
            arrays.extend(layer_grads.values())
        total_norm = np.sqrt(sum(float(np.sum(g * g)) for g in arrays))
        if total_norm <= grad_clip:
            return
        scale = grad_clip / (total_norm + 1e-12)
        grads["emb"] *= scale
        grads["pos_emb"] *= scale
        grads["Wout"] *= scale
        grads["bout"] *= scale
        for layer_grads in grads["layers"]:
            for grad in layer_grads.values():
                grad *= scale

    def _adamw_update(
        self,
        key: str,
        param: np.ndarray,
        grad: np.ndarray,
        lr: float,
        weight_decay: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if key not in self.adam_m:
            self.adam_m[key] = np.zeros_like(param)
            self.adam_v[key] = np.zeros_like(param)
        m = self.adam_m[key]
        v = self.adam_v[key]
        m *= beta1
        m += (1 - beta1) * grad
        v *= beta2
        v += (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1**self.adam_step)
        v_hat = v / (1 - beta2**self.adam_step)
        update = m_hat / (np.sqrt(v_hat) + eps)
        if weight_decay:
            update = update + weight_decay * param
        param -= lr * update

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
                c = context[-self.context_size :]
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
            "num_heads": int(self.num_heads),
            "emb": self.emb.tolist(),
            "pos_emb": self.pos_emb.tolist(),
            "layers": [{name: value.tolist() for name, value in layer.items()} for layer in self.layers],
            "Wout": self.Wout.tolist(),
            "bout": self.bout.tolist(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(weights, f)

    @staticmethod
    def load_weights(filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            weights = json.load(f)
        model = CalculatorModel(
            vocab_size=weights["vocab_size"],
            context_size=weights["context_size"],
            emb_dim=weights["emb_dim"],
            ff_dim=weights["ff_dim"],
            num_layers=weights.get("num_layers", 1),
            num_heads=weights.get("num_heads", 1),
        )
        model.emb = np.array(weights["emb"], dtype=np.float32)
        model.pos_emb = np.array(weights["pos_emb"], dtype=np.float32)
        model.Wout = np.array(weights["Wout"], dtype=np.float32)
        model.bout = np.array(weights["bout"], dtype=np.float32)

        if "layers" in weights:
            model.layers = [
                {name: np.array(value, dtype=np.float32) for name, value in layer.items()}
                for layer in weights["layers"]
            ]
            for layer in model.layers:
                layer.setdefault("ln1_g", np.ones(model.emb_dim, dtype=np.float32))
                layer.setdefault("ln1_b", np.zeros(model.emb_dim, dtype=np.float32))
                layer.setdefault("ln2_g", np.ones(model.emb_dim, dtype=np.float32))
                layer.setdefault("ln2_b", np.zeros(model.emb_dim, dtype=np.float32))
        else:
            layer = {
                "ln1_g": np.ones(model.emb_dim, dtype=np.float32),
                "ln1_b": np.zeros(model.emb_dim, dtype=np.float32),
                "Wq": np.array(weights["Wq"], dtype=np.float32),
                "Wk": np.array(weights["Wk"], dtype=np.float32),
                "Wv": np.array(weights["Wv"], dtype=np.float32),
                "Wo": np.array(weights["Wo"], dtype=np.float32),
                "ln2_g": np.ones(model.emb_dim, dtype=np.float32),
                "ln2_b": np.zeros(model.emb_dim, dtype=np.float32),
                "W1": np.array(weights["W1"], dtype=np.float32),
                "b1": np.array(weights["b1"], dtype=np.float32),
                "W2": np.array(weights["W2"], dtype=np.float32),
                "b2": np.array(weights["b2"], dtype=np.float32),
            }
            model.layers = [
                {name: value.copy() for name, value in layer.items()}
                for _ in range(model.num_layers)
            ]
        model.num_layers = len(model.layers)
        return model
