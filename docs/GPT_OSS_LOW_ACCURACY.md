# gpt-oss-20b scores ~chance on arc_challenge / mmlu — root-cause analysis

## TL;DR

gpt-oss-20b lands at **~random-chance accuracy** (arc_challenge `acc ≈ 0.23`, i.e. 1/4)
across the OpenVINO, llama.cpp, and ONNX Runtime backends. This is **two
independent problems** that happen to produce the same bad number:

| Backend | Model health | Cause of low accuracy |
|---|---|---|
| **ONNX Runtime** (`webgpu-gpt-oss-20b-onnx`) | **Broken** — emits gibberish | int4 conversion of the MoE corrupts the weights |
| **OpenVINO** (`gpt-oss-20b-int4-ov`) | **Healthy** | reasoning-model ↔ multiple-choice-loglikelihood format mismatch |
| **llama.cpp** (`gpt-oss-20b-Q4_K_M.gguf`) | **Healthy** | same format mismatch |

> The harness (`lm_eval`) and the execution providers are **not** the cause.
> Proof: `webgpu-qwen3-4b-onnx`, built with the *identical* `builder.py` flags,
> generates perfectly coherent text and scores normally.

---

## How to read the symptom

arc_challenge is 4-way multiple choice, so `acc ≈ 0.25` is the **random-guess
floor**. Observed:

```
ONNX  arc_challenge : acc 0.2270  acc_norm 0.2577
OV    arc_challenge : acc 0.2346 / 0.2952   (two runs)
```

Both are statistically indistinguishable from guessing. A useful per-sample
diagnostic is the **average continuation log-prob per word**:

```
ONNX avg ll/word : -14.89     <- model maximally "surprised" (≈ uniform logits)
OV   avg ll/word :  -6.02     <- well-calibrated, but choices too close to separate
```

`ln(1/vocab) = ln(1/201088) = -12.2`. ONNX is *below* even that on a per-word
basis — the model is not discriminating at all.

---

## Problem 1 — ONNX Runtime build is genuinely corrupt

### Evidence

1. **Greedy generation is gibberish**, for plain, harmony-chat, and arc-style
   prompts:
   - WebGPU: `' The,\n\ny (p youYou2\n\n-ated\n\n\n\n-.Thely_ Theized1可ine於'`
   - CPU (same model): `' Thise           . \n. \n\n\n\nss ofof (0ed#The (s,/-'`
2. **Same gibberish on both CPU and WebGPU** → not an execution-provider bug.
3. Logits are *finite* (the run used `enableGraphCapture: 0` and completed under
   `non_finite_policy=error` without aborting) but **flat** — per-token log-prob
   ≈ −12 to −18, ≈ `ln(1/vocab)`. So this is **not** the
   [WebGPU all-NaN graph-capture bug](WEBGPU_NAN_BUG.md) (that needs
   `enableGraphCapture: 1`); the logits here are finite-but-wrong.
4. The harness tokenization/alignment is correct (verified: `encode("") == []`,
   `full_tokens` starts with `context_enc`, continuation tokens decode sensibly).
5. **Control: `webgpu-qwen3-4b-onnx`, built with identical flags, works** —
   greedy-generates `"The capital of France is Paris. ..."`. The pipeline and
   harness are fine; the breakage is gpt-oss-20b-specific.

### Build command

```
python builder.py -m openai/gpt-oss-20b -o webgpu-gpt-oss-20b-onnx \
  -c honry-cache-dir -p int4 -e webgpu --extra_options \
  shared_embeddings=true int4_algo_config=rtn_last int4_is_symmetric=true \
  enable_webgpu_graph=true prune_lm_head=true hf_remote=false
```

### Suspected root cause

The int4 quantization of gpt-oss's **MoE** layers. gpt-oss ships **MXFP4**
experts; naive RTN int4-symmetric quantization of the expert/router weights is
the most likely culprit. (`prune_lm_head` / `shared_embeddings` are secondary
suspects but Qwen3-4B used them fine.)

### Fix

1. Re-convert and **validate with a greedy-coherence smoke test before running
   any eval** — if it generates gibberish, don't bother scoring it.
2. Things to try, bisecting:
   - higher precision (fp16) for MoE expert/router layers,
   - non-symmetric int4 (`int4_is_symmetric=false`),
   - drop `prune_lm_head` / `shared_embeddings`,
   - confirm this `builder.py` version actually supports gpt-oss MXFP4 MoE.
3. Baselines to compare against live in the models dir:
   `unsloth/gpt-oss-20b-GGUF` (llama.cpp) and `ov-genai/gpt-oss-20b-int4-ov`
   (OpenVINO) — both are healthy.

---

## Problem 2 — OpenVINO & llama.cpp models are healthy; the *eval format* is wrong

### Evidence the models are fine

- **llama.cpp** (`gpt-oss-20b-Q4_K_M.gguf`) answers correctly:
  ```
  Question: What is the capital of France?
  Answer:
  [Start thinking]
  The user asks ... The answer: Paris. Provide the answer.
  [End thinking]
  Paris.
  ```
- **OpenVINO** per-sample log-probs are well-calibrated (avg −6.0/word) and
  discriminative where the answer is clear — e.g. arc doc with photosynthesis
  options, it strongly prefers *"Light energy is converted to chemical energy"*
  (−17.6) over the distractors (−28 to −34).

### Root cause: reasoning model vs. plain multiple-choice loglikelihood

gpt-oss is a **reasoning model** using the harmony format — it is trained to emit
an `analysis` / `[Start thinking] … [End thinking]` block **before** the final
answer.

lm-eval's MC tasks (arc_challenge, mmlu, hellaswag) score:

```
P( answer_text | "Question: ...\nAnswer:" )
```

i.e. they force each candidate answer to follow `Answer:` **immediately, with no
reasoning step**. The model assigns low and poorly-separated probability to
directly blurting an answer, so the four choices score within ~1 nat of each
other (e.g. arc doc 0: −32.1 to −33.6) → effectively a coin flip → ~chance.

This is a well-known limitation: instruct/reasoning models underperform base
models on raw MC-loglikelihood, and the HF base-model card numbers will not
reproduce this way.

### Fix (for the healthy OV / GGUF models)

- Don't evaluate a reasoning model with bare loglikelihood MC. Instead:
  - use a **generative / CoT** eval — let the model produce its analysis + final
    answer, then extract the chosen letter (e.g. `*_gen`/`*_cot` task variants),
    or
  - use a **chat-templated** MC harness that respects the harmony format,
    rather than the plain `Question:/Answer:` continuation.
- Compare against the **base** model or against **generative** accuracy, not the
  base-model card's loglikelihood-MC numbers.

> Note: the ONNX build has *both* problems, but its weights are corrupt, so fix
> the conversion first; even after that it will still need the format fix to
> match the OV/GGUF behaviour.

---

## Appendix: harness off-by-one fix (correctness, not the gpt-oss cause)

While investigating, a real bug was found and fixed in the ONNX backend's
log-likelihood scoring ([`lm_eval/models/onnxruntime.py`](lm_eval/models/onnxruntime.py)).
The old code derived `context_len` from a BOS-stripped `context_enc` but split a
non-stripped `full_tokens`, so for tokenizers that prepend BOS the last context
token leaked into the continuation (and an empty-context branch dropped the first
continuation token for tokenizers without BOS).

Fixed by computing the split from one consistent tokenization basis:

```python
full_tokens = self.tok_encode(full_text)
context_len = len(self.tok_encode(context)) if context else 0
continuation_tokens = full_tokens[context_len:]
```

This does **not** change the gpt-oss numbers (its weights are corrupt), but it
corrects loglikelihood scoring for healthy models whose tokenizer prepends BOS
(Llama, Qwen, etc.).

---

## Reproduce / diagnose

```bash
# Per-token logit stats; --provider webgpu|cpu. CPU-vs-WebGPU greedy compare
# isolates a broken model from a broken execution provider.
python scripts/diag_logits.py --model <onnx-model-dir> --provider webgpu --dump

# Quick coherence smoke test (llama.cpp): healthy model answers "Paris."
llama-completion -m gpt-oss-20b-Q4_K_M.gguf -n 40 \
  -p $'Question: What is the capital of France?\nAnswer:'
```

Healthy model → coherent text. Broken model → gibberish. Run this **before**
spending hours on a full benchmark.
