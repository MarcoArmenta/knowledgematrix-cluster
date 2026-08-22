# Characterizing LLM alignment behavior with knowledge matrices

**Status: research plan — 2026-08-22.**
This branch (`interpretability`) hosts the alignment benchmarking study. The
computational objects come from the
[knowledgematrix](https://github.com/samueleblanc/knowledgematrix) library;
this repository consumes them at cluster scale. That division of labor is
deliberate and should be preserved:

> **knowledgematrix computes objects with which one studies neural networks.
> The alignment study is a user of those objects, not part of the library.**

Anything of general use (new layer linearizations, extraction modes, model
loaders) is contributed upstream to Samuel's library via PRs; everything
specific to this study (benchmark harness, feature stores, probes, plots)
lives here.

## 1. Goal

Study the behavior of LLMs — first target: **Qwen3.5-4B** — on alignment
benchmarks, and characterize *when the model behaves well versus badly* using
knowledge matrices: for each prompt, an **exact, input-specific decomposition**
of the model's output over its embedded input, `mat.sum(1) == forward` to
machine precision.

The working hypothesis is that knowledge matrices are a sharper instrument
than current mechanistic-interpretability tools because:

1. **Exactness.** The decomposition reproduces the network function exactly at
   the input — no first-order approximation (unlike attribution patching), no
   training of an auxiliary model (unlike sparse autoencoders).
2. **Invariance.** Spectral statistics of knowledge matrices are invariant
   under the network's rescaling/isomorphism symmetries (the quiver-moduli
   point of view of arXiv:2007.12213 / 2109.14589): they are observables of
   the *function*, not of a parametrization — raw hidden activations are not
   (arXiv:2409.13163).
3. **Whole-path composition.** One object composes input→output through every
   layer, instead of per-layer snapshots that must be stitched together.

This hypothesis is to be **tested against baselines, not assumed** (§4,
Stage A).

## 2. The objects and how to compute them

All computed with the upstream library (branch
`claude/qwen3.5-knowledge-matrix-uktaf7`, PR to samueleblanc/knowledgematrix):

| Object | API | Shape | Use |
|---|---|---|---|
| Full matrix | `KnowledgeMatrixComputer(m).forward(x)` | `(T·V, T·D+1)` | small models, validation |
| Hidden-level matrix | build model with `include_lm_head=False` | `(T·D, T·D+1)` | 4B scale; LM head is linear, push rows through after |
| Position blocks | `mat[:, :-1].reshape(T, out, T, D)` | `(T, out, T, D)` | per-token structure |
| Selected rows | `KnowledgeRowComputer(m).forward(x, rows)` | `(k, T·D+1)` | **the workhorse**: ~1 forward pass per row |
| Slopes | either computer with `extract_weff=True` | same, no bias col | causal perturbation tests |

**The two linearization modes** (both exact — row sums reproduce the output):

- `mixer_mode="ratio"` (library default): mixing layers are treated as
  elementwise post/pre ratios. Attribution stays **within each token
  position** — the matrix is block diagonal over positions. Cheap; per-token
  profiles only.
- `mixer_mode="frozen"`: attention weights, DeltaNet recurrence gates, and
  GLU gates are frozen at the actual input; value paths act linearly.
  Attribution **flows across positions** (causally: future blocks are exactly
  zero). This is the mode for "which prompt token drove the decision".

**The main per-prompt object of the study:** the *contrast row*. For a
benchmark item whose good/bad behavior is decided at one next-token position,
let `w = W_lm[good_token] − W_lm[bad_token]`. The decomposition of `w·h_T`
over the `T·D` embedded input coordinates is one row-combination of the
hidden-level matrix — computable directly with `KnowledgeRowComputer` in
frozen mode. It answers: *where in the prompt did the margin between behaving
well and behaving badly come from?*

## 3. Benchmark design

- **Single-token contrasts.** Use benchmarks where behavior is decided at one
  position: refusal sets (next token "Sure" vs "I"), model-written evals
  (yes/no probes for sycophancy, power-seeking, self-preservation), BBQ /
  TruthfulQA-MC (option letter). No multi-step generation needed.
- **Matched pairs.** Harmful/harmless versions of the same template, so
  surface-form confounds cancel in the features.
- **Labels from behavior.** The model's actual choice labels each prompt
  good/bad; we characterize *when* it fails, not whether.
- **Base vs. instruct.** Run Qwen3.5-4B base and instruct on identical
  prompts: same architecture and tokenizer, so *differences* of knowledge
  matrices isolate what alignment training changed.

## 4. Methodology (three stages)

**Stage A — discriminate.** Per prompt, extract low-dimensional
gauge-invariant features: per-token attribution mass and entropy of the
contrast row; singular spectra / stable rank / trace of the decision
position's blocks; angle between the contrast row and the prompt embedding
directions. Train a simple cross-validated probe (logistic / linear) to
predict good vs bad. **Mandatory baselines**: the same probe on raw last-token
hidden states, on logit-lens trajectories, and on attention entropies.
Beating these is the evidence for the hypothesis; not beating them is a
publishable negative that sharpens the theory.

**Stage B — localize.** Where does the discriminative signal live? Attribution
concentrated on the payload vs. the instruction template vs. chat-format
tokens is already a characterization of failure modes (e.g. "failures
co-occur with attribution collapsing onto the jailbreak wrapper").

**Stage C — validate causally.** Rows of `W_eff` are directions in embedding
space, exact within the activation region. Perturb the input embedding along
the contrast row and verify the behavior flips at the predicted magnitude —
and does not flip along matched random directions. This is the test that
distinguishes a real mechanism from a correlate, and where exactness beats
patching approximations.

## 5. What is already built (upstream library)

- [x] Qwen3.5 architecture: `GatedDeltaNet`, `GatedAttention`, `SwiGLU`,
      `Qwen3_5` model, HF weight loader (zero-centered norms folded), parity
      with HF transformers to ~1e-8 (float64).
- [x] `mixer_mode="frozen"` cross-position linearization (exact; causal).
- [x] `KnowledgeRowComputer`: reverse-mode row extraction, rows equal
      forward-mode matrix rows to machine precision, both modes.
- [x] CPU/GPU device parametrization (CUDA test present; **not yet validated
      on real GPU hardware — first cluster job should run the test suite**).

## 6. What needs to be built (this repository)

1. **Sync this fork's `main`** with upstream once the PRs merge (this fork
   currently lags samueleblanc/knowledgematrix).
2. **Benchmark harness** (`alignment/` package):
   - prompt loaders for the chosen benchmarks + chat-template application
     with the Qwen3.5 tokenizer;
   - per-prompt pipeline: forward → behavior label → contrast row (frozen
     mode) → per-token profile + block spectra → one record
     (a few MB max) in a results store (parquet/npz per shard);
   - SLURM array-job scripts (one shard per task), resumable.
3. **Feature/probe layer**: the Stage-A probe with the three baselines,
   cross-validation, permutation tests.
4. **Causal validation runner** (Stage C): embedding perturbations along
   extracted rows, behavior-flip curves.
5. **Memory/perf calibration**: measured cost per prompt of the frozen row
   extraction at 4B (target: seconds/prompt on one GPU; the recurrent
   DeltaNet form is O(T) — a chunked variant upstream is a later
   optimization).
6. **Possible upstream follow-ups** as needs arise: frozen mode for
   `MultiHeadAttention` (classic transformers), selected-rows API for logit
   rows through the LM head, half-precision policies with float64 validation
   gates.

## 7. Sizing (Qwen3.5-4B: D=2560, V=248320)

| Object, per prompt (T tokens) | Size (float32) |
|---|---|
| Contrast row | T·2560 · 4 B ≈ 1 MB at T=100 |
| Hidden-level diagonal blocks | T · 2560² · 4 B ≈ 26 MB·T (store spectra instead) |
| Full hidden-level matrix | (T·2560)² · 4 B ≈ 262 GB at T=100 — **do not store; extract rows** |

The study is designed so the stored objects are rows and spectra —
kilobytes to megabytes per prompt, thousands of prompts on modest storage.

## 8. References

- Armenta, Jodoin, *The Representation Theory of Neural Networks* (2021),
  arXiv:2007.12213
- Armenta, Brüstle, Hassoun, Reineke, *Double framed moduli spaces of quiver
  representations* (2021), arXiv:2109.14589
- Leblanc, Rasolomanana, Armenta, *Hidden Activations Are Not Enough* (2024),
  arXiv:2409.13163
