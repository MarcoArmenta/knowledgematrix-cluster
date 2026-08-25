# Upstream PRs to samueleblanc/knowledgematrix — instructions

**Written 2026-08-22; branch prep already executed remotely the same day —
what's left for you is opening the two PRs on GitHub (§ Step 1 / Step 2) and
deciding what to do with the fork's `main` (§ Step 0).**

Goal: contribute the Qwen3.5 / interpretability work from the fork branch
`claude/qwen3.5-knowledge-matrix-uktaf7` upstream, as agreed in the alignment
study plan (general-purpose library code goes to Samuel's repo; the study
itself lives in `Neural-Networks-Matrices/alignment/`).

## Already done (2026-08-22, remote session)

- [x] Full test suite run on the tip of the Qwen branch (`1cdb0f9`), CPU:
      `extra/tests/qwen3_5.py` 14/14 pass **including the HF-parity oracle
      tests** (transformers 5.15.1), `extra/tests/interpretability.py` 6/6
      (1 CUDA test skipped — no GPU; that's cluster milestone M1).
- [x] PR branch **`qwen3.5-model`** pushed at `463ec3a` (first three commits).
- [x] PR branch **`frozen-rows`** pushed at `1cdb0f9` (all four commits).

## Current state (verified)

- Upstream `samueleblanc/knowledgematrix` `main` is at **`7ad36db`**
  ("Merge pull request #16 … model-resnet152") — it already contains all
  previously merged fork PRs (#1–#16).
- **This fork's `main` has DIVERGED from upstream** — see Step 0.
- Branch `claude/qwen3.5-knowledge-matrix-uktaf7` is exactly upstream
  `main` + 4 commits, in this order:

  | SHA | Commit |
  |---|---|
  | `c869b63` | feat: add Qwen3.5/Qwen3-Next building blocks (GatedDeltaNet, GatedAttention, SwiGLU) |
  | `6379fed` | feat: add Qwen3_5 model with Hugging Face weight loader |
  | `463ec3a` | test: Qwen3.5 parity and knowledge matrix invariant tests; document LLM support |
  | `1cdb0f9` | feat: frozen-routing linearization and reverse-mode row extraction |

  No rebase needed — the branch already applies cleanly onto upstream main.

**Plan: two PRs.** PR 1 = the Qwen3.5 architecture (first three commits);
PR 2 = the interpretability primitives (`mixer_mode="frozen"` +
`KnowledgeRowComputer`, last commit), which depends on PR 1. Splitting keeps
each review focused: PR 1 is "a new model family", PR 2 is "new semantics of
the knowledge matrix computation" — the part that deserves Samuel's closest
look.

## Step 0 — decide what to do with the fork's `main` (needs your judgment)

A fast-forward sync is **not possible**: `main` is not merely behind, it has
diverged. It carries 8 commits of your own that upstream does not have —

    e0c8626  implementation for 1d CNNs
    754404a  device changes
    178cfc4  device to knowledgematrixcomputer
    7a842d3  models with transfer learning
    5c89f7a  allow for more input shapes
    9fe428c  typo
    abe73ea  shape error for mnist1d
    0377a24  residuals module list error when empty

— while upstream `main` is 48 commits ahead (all the merged PRs #1–#16,
including the Qwen3.5 base). Some of your `main` commits are likely
**superseded** upstream (e.g. `178cfc4` device support vs upstream's
`dfe900b` "add device choice for KnowledgeMatrixComputer"; 1D conv handling
vs the merged conv-layers work), others (transfer learning / import-your-own-
pretrained-model) may still be worth an upstream PR of their own.

The PR branches below do NOT depend on `main`, so this can wait. Options:

- **(a) recommended:** preserve the old main as a work branch, then reset
  `main` to upstream:
  ```bash
  git remote add upstream https://github.com/samueleblanc/knowledgematrix.git
  git fetch upstream
  git branch legacy-main origin/main && git push origin legacy-main
  git checkout main && git reset --hard upstream/main
  git push --force-with-lease origin main
  ```
  Then cherry-pick from `legacy-main` whatever is not superseded, as a future
  upstream PR.
- **(b)** merge `upstream/main` into `main` and resolve conflicts by hand
  (they will be substantial in `neural_net.py`).
- **(c)** leave `main` alone for now.

## ⚠ Collision check (2026-08-24) — read before opening PR 1

Upstream has **2 open PRs, both yours**, checked against this plan:

- **#12 GradientMatrixComputer** (new file `gradient_matrix.py`, full matrix
  via jacrev, piecewise-linear nets only): **no conflict** with anything
  here. It is complementary to PR 2's `KnowledgeRowComputer` (full matrix ×
  PL nets vs selected rows × transformer nets); PR 2's description should
  cite it (added below).
- **#17 SwiGLU** (gated product `act(gate_proj)⊙value_proj`, no
  down-projection, `alpha` convex attribution split in the KM): **collides
  with PR 1**, which defines a *different* class of the same name in the
  same file (the full LLaMA FFN treated as an activation layer). Note the
  math: frozen mode's treatment of the GLU product is exactly #17's α=0
  split, so #17 generalizes the Qwen treatment of the product.

**Revised order — do NOT open PR 1 as currently pushed:**

1. Get **#17 merged first** (it is older and has been through review). Ping
   Samuel if it is stalled.
2. Then rework the `qwen3.5-model` branch to **compose** #17's `SwiGLU` +
   a separate down-projection `Linear` for the Qwen FFN (the composition
   #17's docstring explicitly anticipates), instead of defining a second
   SwiGLU class — or, if composition changes KM semantics undesirably for
   ratio mode, rename the Qwen block (`GatedFFN`) and say in the PR why
   both exist. Claude can do this rework once #17's merged form is fixed.
3. #12 can merge any time; it does not interact.

## Step 1 — PR 1: Qwen3.5 model support (HOLD until #17 resolves — see above)

Branch **`qwen3.5-model`** is pushed at `463ec3a` (first three commits) but
needs the SwiGLU rework described in the collision check before opening.
After the rework:

Open a PR **from `MarcoArmenta:qwen3.5-model` into `samueleblanc:main`**.

**Title:** `Add Qwen3.5 model support (GatedDeltaNet, GatedAttention, SwiGLU + HF weight loader)`

**Description (paste):**

> This adds LLM support to the library, targeting Qwen3.5 (same
> architecture family as Qwen3-Next): the goal is computing knowledge
> matrices of a real pretrained 4B model for interpretability studies.
>
> **New building blocks** (`neural_net.py`), faithful to the reference
> implementation in HF transformers (`models/qwen3_5`, Apache-2.0):
> - `GatedDeltaNet` — linear attention: depthwise causal conv + SiLU,
>   per-head gated delta-rule recurrence, SiLU-gated RMS normalization.
> - `GatedAttention` — full attention with fused query+output-gate
>   projection, per-head QK RMSNorm, partial RoPE, GQA, sigmoid output gate.
> - `SwiGLU` — the LLaMA/Qwen-style gated feed-forward.
>
> All three map `d_model -> d_model` per token, so the knowledge matrix
> computation treats them like activations (elementwise post/pre ratio),
> exactly as it already does for `MultiHeadAttention`. The activation
> isinstance tuples of `NN.forward` and `KnowledgeMatrixComputer` are
> unified into a single `ACTIVATION_LAYERS` constant. Also adds
> `NN.identity()`, a no-op boundary layer needed for chained pre-norm
> residuals (`x = x + f(norm(x))`).
>
> **Model + loader** (`models/qwen3_5.py`): `Qwen3_5(NN)` builds the hybrid
> decoder (3 linear-attention layers per full-attention layer, pre-norm
> residual wiring, final RMSNorm, optional LM head).
> `Qwen3_5.from_huggingface` accepts a hub id/path or an already-loaded
> transformers model and copies weights so both compute the same function
> (zero-centered RMSNorm weights folded, tied embeddings resolved).
> `include_lm_head=False` stops at the final hidden states, keeping the
> knowledge matrix tractable at 4B scale (the LM head is linear and can be
> pushed through afterwards). `transformers` stays optional (`[hf]` extra).
>
> **Tests** (`extra/tests/qwen3_5.py`): forward parity against a tiny
> randomly-initialized HF Qwen3.5 in float64 (no weight download; skipped
> without transformers); the knowledge matrix invariant
> `mat.sum(1) == forward` to 1e-8 on hybrid mini models with and without
> the LM head; module validation, shape and causality tests (torch only).
> README gains a "Large language models (Qwen3.5)" section.

## Step 2 — PR 2: frozen-mode linearization + row extraction

Branch **`frozen-rows`** is already pushed at `1cdb0f9` (all four commits).
**Open it after PR 1 merges** (its diff includes PR 1's commits until then).
If you want it visible earlier, open it as a **draft** noting it stacks on
PR 1.

Open a PR **from `MarcoArmenta:frozen-rows` into `samueleblanc:main`**.
(If PR 1 merged, GitHub will show only `1cdb0f9` in the diff. If upstream
main moved, merge it in first — no rebase needed on a shared branch.)

**Title:** `Frozen-routing linearization (mixer_mode="frozen") and reverse-mode row extraction (KnowledgeRowComputer)`

**Description (paste):**

> Two primitives for interpretability studies at LLM scale. Stacks on the
> Qwen3.5 model PR.
>
> **`mixer_mode="frozen"`** (`KnowledgeMatrixComputer`): `GatedAttention`,
> `GatedDeltaNet` and `SwiGLU` gain `frozen_forward(x, x0)` — a
> linearization that freezes the routing (attention weights and output
> gate; recurrence q/k, gates g/beta, conv-SiLU ratio, silu(z) gate and
> output RMS; GLU silu(gate) factor) at the actual layer input `x0` and
> lets the value paths act linearly. The map equals the layer output at
> `x0` exactly, so the row-sum invariant `mat.sum(1) == forward` is
> preserved — but attribution can now flow **across token positions**,
> which the default elementwise ratio treatment (block-diagonal over
> positions) cannot express. Causality is preserved: blocks above the
> diagonal stay exactly zero. The default `"ratio"` mode is byte-unchanged.
>
> **`KnowledgeRowComputer`** (`row_computer.py`): computes selected rows of
> the knowledge matrix (or W_eff) by reverse mode — one vector–Jacobian
> product per row over the frozen linear graph, ≈ one forward pass per row
> independent of input size. This is the efficient direction when a few
> output rows (an answer token's logits) are needed over many input
> coordinates; a full matrix is still cheaper via `KnowledgeMatrixComputer`.
> Rows agree with `KnowledgeMatrixComputer` to machine precision in both
> modes (tested). Scope: transformer-style models; conv/pool models raise
> `NotImplementedError`.
>
> Complementary to #12 (`GradientMatrixComputer`): that PR computes the
> **full** matrix as gradient×input for **piecewise-linear** nets (where the
> Jacobian and the KM slopes coincide); this one computes **selected rows**
> for **transformer-style** nets over the frozen linear graph (where they
> don't — the frozen map excludes gate derivatives by design, preserving
> the row-sum invariant instead).
>
> Motivation: this is the workhorse of an alignment-benchmarking study on
> Qwen3.5-4B (per-prompt "contrast rows" decomposing the margin between
> aligned and misaligned next tokens over the embedded input) — happy to
> share details.

## Step 2b — PR 3: frozen mode for MultiHeadAttention (stacked on PR 2)

Branch **`frozen-mha`** is pushed at `6328b43` (one commit on top of
`frozen-rows`). Open **after PR 2 merges** (until then its diff shows the
whole stack), from `MarcoArmenta:frozen-mha` into `samueleblanc:main`.

**Title:** `Frozen-routing linearization for MultiHeadAttention`

**Description (paste):**

> Extends `mixer_mode="frozen"` to classic softmax attention:
> `MultiHeadAttention` gains `frozen_forward(x, x0)` mirroring
> `GatedAttention` — the softmax attention weights (with GQA/MQA and the
> optional mask) are computed from and frozen at the actual layer input,
> and `x` flows linearly through the value/output projections. Equals
> `forward(x0)` at `x = x0` exactly, so the row-sum invariant is
> preserved; classic transformers now get **cross-position attribution**
> under frozen mode and `KnowledgeRowComputer`, instead of the
> block-diagonal ratio treatment. The default `"ratio"` mode is
> byte-unchanged (regression-tested).
>
> Tests (`extra/tests/frozen_mha.py`, torch-only, float64): exactness at
> x0 incl. GQA/MQA and causal mask; linearity of the frozen map; exact
> zeros above the diagonal under a causal mask and nonzero cross-position
> blocks without one; Transformer row-sum invariant in frozen mode;
> row/matrix agreement; ratio-mode regression.

## Step 3 — after both merge

1. Sync fork `main` with upstream (after resolving Step 0 this becomes a
   plain fast-forward).
2. In `Neural-Networks-Matrices`, point `alignment/requirements.txt` at
   upstream instead of the fork branch:
   `knowledgematrix @ git+https://github.com/samueleblanc/knowledgematrix`.
3. Optionally delete the fork branches `qwen3.5-model` and `frozen-rows`;
   keep `claude/qwen3.5-knowledge-matrix-uktaf7` until then.

## Later upstream candidates (not now)

From the alignment study, once proven there (see
`Neural-Networks-Matrices/alignment/RESEARCH_PLAN.md` §6):
- a row-*combination* API (`w·rows` in one backward — the study's
  `ContrastRowComputer` subclass is the prototype);
- exposing attention weights for entropy baselines;
- frozen mode for classic `MultiHeadAttention`;
- half-precision policies with float64 validation gates; chunked DeltaNet.
