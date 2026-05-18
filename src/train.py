import argparse
import json
import os
import sys
import time
from pathlib import Path

# Keep this before importing numpy. BLAS/OpenMP libraries usually read these
# values at import time, so setting them later may not affect the process.
DEFAULT_NUM_THREADS = "4"
os.environ.setdefault("OPENBLAS_NUM_THREADS", DEFAULT_NUM_THREADS)
os.environ.setdefault("OMP_NUM_THREADS", DEFAULT_NUM_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", DEFAULT_NUM_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", DEFAULT_NUM_THREADS)

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data import (
    DEFAULT_TRAIN_OPERATORS,
    DIGITS,
    CalculatorTokenizer,
    build_training_data,
    generate_corpus,
    split_corpus,
)
from src.model import CalculatorModel


def train_epoch(
    model: CalculatorModel,
    contexts: np.ndarray,
    targets: np.ndarray,
    lr: float,
    batch_size: int,
    grad_clip: float,
    weight_decay: float,
    epoch: int,
    epochs: int,
    log_every_batches: int,
) -> float:
    n = contexts.shape[0]
    order = np.random.permutation(n)
    total_loss = 0.0
    batch_count = (n + batch_size - 1) // batch_size
    epoch_start = time.time()

    for batch_num, start in enumerate(range(0, n, batch_size), start=1):
        batch = order[start : start + batch_size]
        loss, grads = model.loss_and_gradients(contexts[batch], targets[batch])
        model.update(grads, lr=lr, grad_clip=grad_clip, weight_decay=weight_decay)
        total_loss += loss * len(batch)

        should_log = (
            log_every_batches > 0
            and (batch_num == 1 or batch_num == batch_count or batch_num % log_every_batches == 0)
        )
        if should_log:
            seen = min(start + batch_size, n)
            avg_loss = total_loss / seen
            elapsed = time.time() - epoch_start
            batches_per_sec = batch_num / max(elapsed, 1e-9)
            print(
                f"  epoch {epoch}/{epochs} "
                f"batch {batch_num}/{batch_count} "
                f"tokens {seen}/{n} "
                f"avg_loss={avg_loss:.4f} "
                f"speed={batches_per_sec:.2f} batches/s",
                flush=True,
            )

    return total_loss / n


def token_accuracy(model: CalculatorModel, contexts: np.ndarray, targets: np.ndarray) -> float:
    correct = 0
    for start in range(0, len(targets), 512):
        ctx = contexts[start : start + 512]
        _, logits, _ = model.forward(ctx)
        pred = np.argmax(logits, axis=1)
        correct += int(np.sum(pred == targets[start : start + 512]))
    return correct / max(1, len(targets))


def generate_rhs(
    model: CalculatorModel,
    tokenizer: CalculatorTokenizer,
    prompt: str,
    max_new_tokens: int = 24,
) -> str:
    context = list(tokenizer.encode_text(prompt, add_eos=False))
    generated: list[int] = []
    digit_ids = [tokenizer.token_to_idx[digit] for digit in DIGITS]
    minus_id = tokenizer.token_to_idx["-"]

    for _ in range(max_new_tokens):
        if len(context) < model.context_size:
            padded = [tokenizer.pad_id] * (model.context_size - len(context)) + context
        else:
            padded = context[-model.context_size :]

        dist = model.predict_distribution(np.array(padded, dtype=np.int32))
        allowed_ids = digit_ids + [tokenizer.eos_id]
        if not generated:
            allowed_ids.append(minus_id)
        masked = np.zeros_like(dist)
        masked[allowed_ids] = dist[allowed_ids]
        if masked.sum() > 0:
            dist = masked / masked.sum()
        next_id = int(np.argmax(dist))
        if next_id == tokenizer.eos_id:
            break
        generated.append(next_id)
        context.append(next_id)

    return "".join(tokenizer.decode(generated))


def exact_match_accuracy(
    model: CalculatorModel,
    tokenizer: CalculatorTokenizer,
    corpus: list[str],
) -> float:
    if not corpus:
        return 0.0

    matches = 0
    for sample in corpus:
        lhs, rhs = sample.split("=", 1)
        pred = generate_rhs(model, tokenizer, f"{lhs}=", max_new_tokens=len(rhs) + 4)
        if pred == rhs:
            matches += 1
    return matches / len(corpus)


def save_training_params(path: Path, args: argparse.Namespace, metrics: dict) -> None:
    params = vars(args).copy()
    params["metrics"] = metrics
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the toy LLM calculator model")
    parser.add_argument("--num-expr", type=int, default=30603)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--context-size", type=int, default=32)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--min-value", type=int, default=0)
    parser.add_argument("--max-value", type=int, default=100)
    parser.add_argument("--operators", type=str, default=DEFAULT_TRAIN_OPERATORS)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, default="weight")
    parser.add_argument("--name", type=str, default="calculator_transformer")
    parser.add_argument("--eval-expr", type=int, default=100)
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=25,
        help="Print training progress every N batches. Use 0 to disable batch logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / f"{args.name}.json"
    tokenizer_path = output_dir / f"{args.name}_tokenizer.json"
    params_path = output_dir / f"{args.name}_params.json"

    print("Training configuration:", flush=True)
    print(f"  expressions={args.num_expr}", flush=True)
    print(f"  epochs={args.epochs}", flush=True)
    print(f"  batch_size={args.batch_size}", flush=True)
    print(f"  context_size={args.context_size}", flush=True)
    print(f"  number_range={args.min_value}..{args.max_value}", flush=True)
    print(f"  operators={args.operators}", flush=True)
    print(
        f"  model emb_dim={args.emb_dim} ff_dim={args.ff_dim} "
        f"layers={args.num_layers} heads={args.num_heads}",
        flush=True,
    )
    print(f"  optimizer=AdamW weight_decay={args.weight_decay}", flush=True)
    print(f"  output_dir={output_dir}", flush=True)
    print("  cpu math threads:", flush=True)
    print(f"    OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}", flush=True)
    print(f"    OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    print(f"    MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}", flush=True)
    print(f"    NUMEXPR_NUM_THREADS={os.environ.get('NUMEXPR_NUM_THREADS')}", flush=True)

    print("Generating integer calculator corpus...", flush=True)
    corpus = generate_corpus(
        num_expr=args.num_expr,
        min_value=args.min_value,
        max_value=args.max_value,
        operators=args.operators,
        seed=args.seed,
    )
    print(f"Generated {len(corpus)} equations.", flush=True)
    print(f"Example equations: {corpus[:5]}", flush=True)

    print("Splitting train/validation corpus...", flush=True)
    train_corpus, val_corpus = split_corpus(corpus, args.validation_fraction, seed=args.seed)

    tokenizer = CalculatorTokenizer()
    print("Building train token dataset...", flush=True)
    train_contexts, train_targets = build_training_data(
        train_corpus, tokenizer, args.context_size
    )
    print("Building validation token dataset...", flush=True)
    val_contexts, val_targets = build_training_data(val_corpus, tokenizer, args.context_size)

    print(f"Vocabulary: {tokenizer.token_list}", flush=True)
    print(f"Train samples: {len(train_corpus)} equations, {len(train_targets)} tokens", flush=True)
    print(f"Validation samples: {len(val_corpus)} equations, {len(val_targets)} tokens", flush=True)

    print("Initializing model...", flush=True)
    model = CalculatorModel(
        vocab_size=len(tokenizer.token_list),
        context_size=args.context_size,
        emb_dim=args.emb_dim,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        seed=args.seed,
    )

    metrics: dict[str, float | int] = {}
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"Starting epoch {epoch}/{args.epochs}...", flush=True)
        epoch_start = time.time()
        loss = train_epoch(
            model,
            train_contexts,
            train_targets,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_clip=args.grad_clip,
            weight_decay=args.weight_decay,
            epoch=epoch,
            epochs=args.epochs,
            log_every_batches=args.log_every_batches,
        )
        epoch_seconds = time.time() - epoch_start
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print("  running validation token accuracy...", flush=True)
            if len(val_targets) > 0:
                val_acc = token_accuracy(model, val_contexts, val_targets)
                val_text = f"val_token_acc={val_acc * 100:.2f}% "
            else:
                val_acc = None
                val_text = "val_token_acc=skipped "
            print(
                f"Epoch {epoch:3d}/{args.epochs} "
                f"loss={loss:.4f} {val_text}"
                f"time={epoch_seconds:.1f}s",
                flush=True,
            )
            metrics["last_loss"] = float(loss)
            if val_acc is not None:
                metrics["last_val_token_accuracy"] = float(val_acc)
        else:
            print(
                f"Epoch {epoch:3d}/{args.epochs} loss={loss:.4f} time={epoch_seconds:.1f}s",
                flush=True,
            )

    print("Generating evaluation corpus...", flush=True)
    eval_corpus = generate_corpus(
        num_expr=args.eval_expr,
        min_value=args.min_value,
        max_value=args.max_value,
        operators=args.operators,
        seed=args.seed + 1,
    )
    metrics["eval_exact_match_accuracy"] = float(
        exact_match_accuracy(model, tokenizer, eval_corpus)
    )
    metrics["elapsed_seconds"] = round(time.time() - start_time, 3)

    print("Saving model artifacts...", flush=True)
    model.save_weights(str(weights_path))
    tokenizer.save(str(tokenizer_path))
    save_training_params(params_path, args, metrics)

    print(f"Saved weights: {weights_path}")
    print(f"Saved tokenizer: {tokenizer_path}")
    print(f"Saved training parameters: {params_path}")
    print(f"Eval exact-match accuracy: {metrics['eval_exact_match_accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
