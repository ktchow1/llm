Below is a clean, end-to-end **specification document** of your idea, rewritten from scratch so it can be shared with humans or used directly by another LLM/agent to implement the experiment.

---

# Toy LLM Calculator Experiment — Specification

## 1. Purpose / Intuition

The goal of this project is to build a **minimal transformer-based next-token predictor** to understand:

> Why predicting the next token in a sequence is sufficient to learn structured logic (e.g., arithmetic), even without explicit rules.

We want to experimentally demonstrate:

* A model trained only on next-token prediction
* Can learn implicit rules of arithmetic
* Without being explicitly programmed with math logic

This is a **controlled toy analogue of how LLMs like ChatGPT learn reasoning-like behavior from text alone**.

---

# 2. Core Hypothesis

A transformer trained with:

> next-token prediction on structured sequences

can learn:

* arithmetic rules
* grammar structure
* algorithm-like behavior

purely from statistical patterns in data.

---

# 3. Task Definition

We define a **sequence-to-sequence completion task**, but trained as next-token prediction.

Each training sample is a full arithmetic expression:

```text
A op B = R <EOS>
```

The model learns to predict each next token in the sequence.

---

# 4. Number System Constraints

## Operand range:

```
A, B ∈ [-10000, 10000]
```

## Representation rules:

* Base-10 integers only
* Negative numbers allowed (`-` is part of number)
* No floating point
* No scientific notation

---

# 5. Operators

```
op ∈ { +, -, *, / }
```

## Division rule:

```
A / B = floor(A / B)
```

(no remainder, no decimals)

---

# 6. Tokenization

We use **character-level tokenization**:

### Vocabulary:

```
0 1 2 3 4 5 6 7 8 9
+ - * / =
<EOS>
```

### Important rule:

* Each digit is one token
* Negative sign `-` is part of number token stream, not a separate operation

Example:

```
-123+45=...
```

becomes:

```
- 1 2 3 + 4 5 = ...
```

---

# 7. Dataset Construction

## 7.1 Generation rule

Each sample is generated deterministically:

```
A op B = R <EOS>
```

where:

* A, B are randomly sampled integers in range
* R is computed using exact arithmetic rules defined above

---

## 7.2 Example samples

```
1+2=3<EOS>
12+34=46<EOS>
-5+7=2<EOS>
10/4=2<EOS>
```

---

## 7.3 Critical constraint (IMPORTANT)

Each prefix must have **exactly one correct continuation**.

We do NOT include ambiguous mappings such as:

❌ WRONG:

```
1+2 → =
1+2 → 3
```

✔ CORRECT:
Only full deterministic sequences exist.

---

## 7.4 Optional curriculum (recommended)

Start simple, then increase difficulty:

### Stage 1:

```
A op B (small numbers)
```

### Stage 2:

```
include negatives
```

### Stage 3:

```
expand range to [-10000, 10000]
```

---

# 8. Training Objective

Standard autoregressive language modeling:

```
maximize P(token_t | token_1 ... token_{t-1})
```

Loss:

```
cross entropy loss over next token prediction
```

---

# 9. Model Architecture (toy-scale)

Designed for CPU training.

| Component      | Value      |
| -------------- | ---------- |
| Layers         | 2          |
| Heads          | 2–4        |
| d_model        | 64         |
| FFN size       | 128        |
| Context length | 28–32      |
| Parameters     | ~100k–300k |

---

# 10. Context Length

Must support full sequence:

```
A op B = R <EOS>
```

Recommended:

```
context_size = 28 (safe baseline)
context_size = 32 (robust choice)
```

---

# 11. Training Behavior

The model is NOT trained to directly compute arithmetic.

Instead it learns:

* structural patterns in sequences
* token transitions in valid equations
* implicit algorithm-like behavior from data regularities

---

# 12. Expected Learned Behavior

After training:

## Prefix behavior examples

```
1+2 → =
1+2= → 3
1+ → digit (building operand)
12 → operator likely (+ - * /)
```

The model learns a **syntactic + arithmetic state machine implicitly**.

---

# 13. Key Conceptual Insight

This experiment demonstrates:

> A next-token predictor does not memorize equations — it learns a compressed model of the data distribution that can behave like an algorithm.

Arithmetic “emerges” because:

* dataset is consistent
* structure is repetitive
* constraints enforce determinism
* model learns transitions, not rules explicitly

---

# 14. What this experiment teaches

This toy system demonstrates:

### 1. Language modeling = structure learning

Not just text generation.

### 2. Logic emerges from distribution

Not from explicit rules.

### 3. Deterministic systems become learnable patterns

When encoded as sequences.

---

# 15. Extension Path (future work)

This system can be extended gradually:

### Phase 1:

Single operator arithmetic (current spec)

### Phase 2:

Multi-operator expressions:

```
1+2+3=
```

### Phase 3:

Operator precedence:

```
1+2*3=
```

### Phase 4:

Decimals and fixed-point arithmetic

---

# 16. Summary

This project builds a minimal environment where:

> A transformer trained only on next-token prediction learns to behave like a calculator.

It serves as a controlled demonstration of:

* why LLMs can learn reasoning-like behavior
* how structure emerges from data
* how probability modeling becomes algorithmic behavior

---

If you want, I can next turn this into a **fully runnable PyTorch project (dataset + model + training + inference script)** in a single clean file.



python3 -m src.train \
  --min-value 0 \
  --max-value 100 \
  --operators "+-*" \
  --num-expr 30603 \
  --validation-fraction 0.1 \
  --epochs 100 \
  --batch-size 256 \
  --context-size 32 \
  --emb-dim 128 \
  --ff-dim 512 \
  --num-layers 4 \
  --num-heads 4 \
  --lr 0.003 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --name calculator_0_100_enhanced

