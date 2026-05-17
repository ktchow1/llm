Below is a **clean, structured Markdown summary** of everything we discussed, written to be **precise, numeric, and implementation-oriented**, matching your intent to *really understand transformers / GPT from first principles*.

You can save this directly as `attention_summary.md`.

---

# Transformer Attention — Precise Summary

## 1. Tokens and Embeddings

* Each token `t` is mapped to a **token embedding vector**
  [
  e_t \in \mathbb{R}^{d_{\text{model}}}
  ]

* In GPT-style models:

  * `d_model` is fixed (e.g. GPT-3: `d_model = 12288`)
  * **Token embeddings are fixed after training**
  * Context dependence comes **later**, via attention, not from embeddings

* The *same token* always starts with the *same embedding*, regardless of context.

---

## 2. Layers vs Heads (NOT the same thing)

* **Layer**: a sequential depth step
* **Head**: a parallel attention subspace *within* a layer

Example (GPT-3 175B):

* Layers: `96`
* Heads per layer: `96`
* This is **not a rule**, just a design choice.

---

## 3. Dimensions (Concrete Numbers)

Let:

* `d_model = 12288`
* `n_heads = 96`
* Then:
  [
  d_{\text{head}} = \frac{d_{\text{model}}}{n_{\text{heads}}} = 128
  ]

---

## 4. Q, K, V Matrices (Per Layer, Per Head)

For **each layer** and **each head**, we have **separate learned matrices**:

[
W_Q^h \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}
]
[
W_K^h \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}
]
[
W_V^h \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}
]

Important:

* **Same matrix is used for all tokens**
* **Different head ⇒ different matrices**
* **Different layer ⇒ different matrices**

---

## 5. Query, Key, Value Vectors (Per Token)

For token `i` with embedding `e_i`:

[
q_i^h = e_i W_Q^h \in \mathbb{R}^{d_{\text{head}}}
]
[
k_i^h = e_i W_K^h \in \mathbb{R}^{d_{\text{head}}}
]
[
v_i^h = e_i W_V^h \in \mathbb{R}^{d_{\text{head}}}
]

So:

* Query dimension = Key dimension = `d_head`
* Value dimension = `d_head`
* A token has **96 different queries per layer** (one per head)

---

## 6. Are Q/K/V Values Bounded?

* Elements of Q, K, V:

  * Type: float (FP16 / BF16 / FP32)
  * **No fixed range**
  * Typically centered around `~N(0, σ²)` after training

* Dot product:
  [
  q_i \cdot k_j
  ]

  * **Unbounded**
  * Can be large or small

---

## 7. Why Scaling by √d?

Scaled dot product attention:

[
\text{score}*{ij} = \frac{q_i \cdot k_j}{\sqrt{d*{\text{head}}}}
]

Reason:

* Prevents dot products from growing too large
* Keeps softmax gradients stable

---

## 8. What Does “Similarity” Mean?

There is **NO absolute threshold** like:

> “If QK > X then similar”

Instead:

* Similarity is **relative**
* Only meaningful **after softmax**

---

## 9. Softmax (Key Property)

[
\text{softmax}(s_i) = \frac{e^{s_i}}{\sum_j e^{s_j}}
]

Effects:

* **Amplifies winners**
* **Suppresses losers**
* Turns raw scores into a probability distribution

Your statement was exactly right:

> softmax amplifies differences among winners but shrinks differences among losers

---

## 10. Directionality of Similarity

Important asymmetry:

[
q_n \cdot k_m ;\text{large} ;;\nRightarrow;; q_m \cdot k_n ;\text{large}
]

Reason:

* `q` and `k` come from **different projections**
* Attention is **directed**

---

## 11. Which Tokens Are Compared?

For **token i attending**:

* Query: `q_i`
* Keys: `{k_1, k_2, ..., k_i}` (causal mask in GPT)
* Yes:

  * Later token compares its query with **earlier keys**
  * Earlier tokens **cannot see later tokens**

---

## 12. Value Vectors

* Value vector of token `j`:

  * **Fixed for that token in that layer & head**
  * Independent of *who* is attending to it

Attention output:

[
\text{output}*i^h = \sum_j \text{softmax}*{ij} \cdot v_j
]

---

## 13. Why Multiple Heads Don’t Collapse

Your concern:

> Heads don’t communicate — why don’t they learn identical things?

Reasons they *don’t* collapse:

1. **Different parameter initialization**
2. **Different gradient signals**
3. **Different subspaces (`W_Q^h, W_K^h, W_V^h`)**
4. **Output projection mixes heads**, so redundancy is penalized
5. Empirically, heads specialize (syntax, position, induction, etc.)

No explicit constraint enforces diversity — it **emerges from optimization pressure**.

---

## 14. Mental Model (Correct)

* A token:

  * Has **one embedding**
  * Produces **96 different queries per layer**
* Each head:

  * Looks at the sequence through a different linear lens
* Layers:

  * Repeatedly remix information, building abstraction depth

---

## 15. One-Sentence Summary

> A transformer layer projects each token embedding into multiple query–key–value subspaces; each head performs directed, relative similarity matching via scaled dot-product + softmax, and layers stack these operations to build context-dependent representations.

---

If you want next:

* **Toy numerical example (with real numbers)**
* **Why induction heads emerge**
* **Full forward pass pseudocode**
* **How this becomes a next-token predictor**

Just say which.
