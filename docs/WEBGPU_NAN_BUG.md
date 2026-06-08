# Bug Report: WebGPU EP returns all-NaN logits during teacher-forced scoring (onnxruntime-genai + `enableGraphCapture`)

## Summary

When scoring multiple-choice answers with **teacher-forced log-likelihood** through
`onnxruntime-genai` on the **WebGPU execution provider**, `Generator.get_logits()`
intermittently returns a logits vector whose values are **entirely non-finite**
(all `151936` vocabulary entries are `NaN`).

The failure is:

- **Non-deterministic** — which `(prompt, answer)` pairs go NaN changes between runs.
- **State-dependent** — the *same* pair can be NaN when scored "cold" and finite
  after the model has processed other requests, and vice-versa.
- **Independent of the input text** — within a single question, some answer
  choices score finite while others (sharing the identical prompt) go NaN.
- **Eliminated by disabling WebGPU graph capture** — removing
  `"enableGraphCapture": "1"` from `genai_config.json` produces **0 NaN** over the
  same inputs.

Together these point to a **WebGPU graph-capture / KV-cache buffer-aliasing
defect**, not a numerical edge case in the model or a bad input.

A standalone reproducer with no heavy dependencies is provided:
[`scripts/repro_webgpu_nan.py`](scripts/repro_webgpu_nan.py) +
[`scripts/webgpu_nan_questions.json`](scripts/webgpu_nan_questions.json).

---

## Environment

| Component | Version |
|---|---|
| OS | Windows 11 (10.0.26200) |
| GPU | Intel(R) UHD Graphics 770 (integrated) |
| Python | 3.12.10 |
| `onnxruntime-genai` | 0.14.1 |
| `onnxruntime-webgpu` | 1.25.1 |
| `onnxruntime` | 1.26.0 |
| `numpy` | 2.4.6 |
| Model | DeepSeek-R1-Distill-Qwen-1.5B, exported to ONNX (qwen2, vocab 151936, 28 layers) |
| Execution provider | `WebGpuExecutionProvider` (Dawn) |

> Note: `onnxruntime-webgpu` must be installed **last** with `--no-deps` so its
> WebGPU-enabled binaries override the plain `onnxruntime` that `onnxruntime-genai`
> pulls in. Verify with
> `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`
> → `['WebGpuExecutionProvider', 'CPUExecutionProvider']`.

### Relevant `genai_config.json`

```json
{
  "model": {
    "decoder": {
      "session_options": {
        "provider_options": [
          { "webgpu": { "enableGraphCapture": "1" } }
        ]
      }
    }
  },
  "search": {
    "past_present_share_buffer": true
  }
}
```

The two interacting settings are **`enableGraphCapture: "1"`** and
**`past_present_share_buffer: true`**.

---

## The access pattern that triggers it

The scoring code computes log-likelihood with **teacher forcing**: append the
prompt once, then for each gold continuation token read the next-token logits and
feed the gold token back in.

```python
params = og.GeneratorParams(model)
params.set_search_options(max_length=4096, do_sample=False)
gen = og.Generator(model, params)

gen.append_tokens(context_tokens)          # the prompt, appended once
for tok in continuation_tokens:            # gold answer tokens, one at a time
    logits = gen.get_logits()              # <-- intermittently ALL NaN
    log_prob = logits[tok] - logsumexp(logits)
    gen.append_tokens([tok])               # teacher forcing: feed gold token
```

Each answer choice uses a **fresh `Generator`** but the **same shared `Model`**.

---

## How to reproduce

```bash
pip install numpy onnxruntime-genai onnxruntime-webgpu
# ensure onnxruntime-webgpu wins (see note above):
pip install --force-reinstall --no-deps onnxruntime-webgpu

# reproduce the NaN
python scripts/repro_webgpu_nan.py --model /path/to/onnx/model-dir

# control: same inputs, graph capture disabled -> no NaN
python scripts/repro_webgpu_nan.py --model /path/to/onnx/model-dir --no-graph-capture
```

The reproducer embeds 12 real questions (captured from an `arc_challenge` run) and
scores all four choices of each. It depends only on `numpy` + `onnxruntime-genai`
+ `onnxruntime-webgpu` — no `lm-evaluation-harness`, `torch`, or `datasets`.

### Observed output (graph capture ON)

```
[env] onnxruntime-genai 0.14.1
[env] onnxruntime providers: ['WebGpuExecutionProvider', 'CPUExecutionProvider']
Q0: Question: At which temperature does water freeze? Answer:
    choice 0:   -8.800  ' 0 degrees Celsius'   (recorded: ok)
    choice 1:  -72.371  ' 32 degrees Celsius'  (recorded: NaN)
    ...
Q3: Question: Cells take in food for energy ...
    choice 0:  -32.339  ' building proteins'
    choice 1: NaN       ' breaking down wastes'        <-- all-NaN logits
    ...
=== Summary ===
  choices scored : 48
  NaN choices    : 1
  -> BUG REPRODUCED: WebGPU returned all-NaN logits.
```

### Observed output (graph capture OFF — `--no-graph-capture`)

```
[setup] enableGraphCapture DISABLED via temp config: ...
=== Summary ===
  NaN choices    : 0
  -> No NaN with graph capture disabled (confirms the trigger).
```

---

## Evidence that this is a graph-capture/buffer bug (not the input)

These experiments were run directly against `onnxruntime-genai` (no harness).

### 1. Same prompt, different choices disagree

In a real `arc_challenge` run, **340 of 1172** documents contained at least one
NaN, and **771 of 4687** individual `(prompt, choice)` scorings were NaN. Within a
single question the prompt is identical across all four choices, yet some score
finite and others NaN:

```
Question: At which temperature does water freeze?
  ' 0 degrees Celsius'    -> finite
  ' 32 degrees Celsius'   -> NaN
  ' 100 degrees Celsius'  -> finite
  ' 212 degrees Celsius'  -> NaN
```

If the prompt were the cause, all four would fail together. They don't.

### 2. The same pair is NaN cold but finite when warmed

Taking the first failing pair (flattened request index **61**, the
`' 32 degrees Celsius'` continuation):

| Scenario | Result |
|---|---|
| Scored as the 62nd request through a shared model | **NaN** |
| Scored **alone** right after another model ran in the same process | **NaN** |
| Scored **after** replaying its 61 predecessors to warm the model | **finite** |
| Scored as the very first thing in a fresh process | **finite** |

The text and tokens are identical in every case — only the GPU/session state
differs. The outcome tracks the state, not the input.

### 3. Reproduced NaN positions don't match recorded positions

Re-running the embedded questions flips *which* choices go NaN relative to the
originally recorded `nan_flags`. The defect is **non-deterministic**.

### 4. Disabling graph capture removes all NaN

Rewriting `genai_config.json` to drop `enableGraphCapture` (everything else
unchanged, same model, same WebGPU EP) yields **0 NaN over all 48 scored
choices**. This isolates the trigger to WebGPU **graph capture**.

### 5. Sequence length is not the discriminator

Token-length distributions of NaN vs. finite scorings fully overlap
(NaN continuation length mean 6.2, finite 6.1; both span 1–25+ tokens), ruling
out a simple "too long / too short" shape issue at the API level.

---

## Suspected root cause

WebGPU **graph capture** records the GPU command graph for a fixed set of input
buffers/shapes and replays it. Combined with **`past_present_share_buffer: true`**
(where present/past KV tensors alias the same backing buffer), replaying a
captured graph against a generator whose KV-cache buffer state differs from
capture time appears to read/write stale or mis-aliased GPU memory, yielding a
fully corrupted (all-NaN) logits output. The non-determinism and state-dependence
are consistent with uninitialized/aliased buffer reuse rather than a math error
(e.g. overflow), which would be deterministic and input-tied.

---

## Impact

- Any benchmark/eval that scores via per-token `get_logits()` teacher forcing
  (log-likelihood tasks: `arc_challenge`, `hellaswag`, `mmlu`, etc.) produces
  **silently wrong** results on WebGPU unless a non-finite guard catches it.
- With a guard that errors on non-finite logits, the run **aborts**; without one,
  NaN log-probs corrupt accuracy metrics.

---

## Workaround

Remove WebGPU graph capture from the model's `genai_config.json`:

```diff
 "provider_options": [
-  { "webgpu": { "enableGraphCapture": "1" } }
+  { "webgpu": {} }
 ]
```

This restores finite logits at a modest performance cost. (The provided
reproducer's `--no-graph-capture` flag does this in a temporary copy without
modifying the original model directory.)

---

## Attachments

- [`scripts/repro_webgpu_nan.py`](scripts/repro_webgpu_nan.py) — standalone reproducer.
- [`scripts/webgpu_nan_questions.json`](scripts/webgpu_nan_questions.json) — 12 captured
  questions with per-choice `nan_flags` from the original run.
