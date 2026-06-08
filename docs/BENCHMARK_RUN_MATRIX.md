# Accuracy benchmark run matrix

Models under test (across OpenVINO / llama.cpp / ONNX Runtime backends):

| Short name | Type | Notes |
|---|---|---|
| Llama-3.2-3B-Instruct | standard instruct | well-suited to LL-MC |
| Phi-4-Mini-Instruct | standard instruct | well-suited to LL-MC |
| Qwen3-4B | hybrid (thinking mode) | LL-MC OK if thinking disabled / chat-templated |
| DeepSeek-R1-Distill-Qwen-1.5B | reasoning (always CoT) | LL-MC underrates it — needs generative |
| GPT-OSS-20B | reasoning (harmony) | needs generative; **ONNX build is corrupt** — see [GPT_OSS_LOW_ACCURACY.md](GPT_OSS_LOW_ACCURACY.md) |

## Why two task tiers

`arc_challenge`, `hellaswag`, `winogrande`, `mmlu` are **loglikelihood multiple-choice**
(score `P(answer | context)`, no generation). They systematically **underrate
reasoning models**, which are trained to emit a thinking block before answering —
the model assigns low, poorly-separated probability to blurting an answer
directly, so the choices land within ~1 nat of each other ≈ chance. (This is
exactly what we observed for healthy gpt-oss on OV/GGUF.)

So we run a **generative tier** (`gsm8k`, `ifeval`) where reasoning models can
actually use their CoT and instruction-following.

## Tasks

| Tier | Tasks | Kind | Run for |
|---|---|---|---|
| 1 — knowledge / commonsense | `arc_challenge`, `hellaswag`, `winogrande`, `mmlu` | loglikelihood-MC | **all models** (baseline / floor) |
| 2 — generative | `gsm8k_cot_zeroshot`, `ifeval` | generation | **all models** (headline for reasoning models) |

> Treat Tier-1 as a *floor* for the reasoning models (R1-Distill, gpt-oss), not
> their headline score. Report Tier-2 as the primary number for them.

Standard few-shot conventions (lm-eval defaults, set per task if you want to
match leaderboards): arc_challenge 25-shot, hellaswag 10-shot, winogrande 5-shot,
mmlu 5-shot, gsm8k 5-shot (or 0-shot CoT), ifeval 0-shot. The runner uses task
defaults unless you pass `--num-fewshot`.

## What each benchmark measures & why it's recommended

### How "loglikelihood-MC" vs "generative" scoring works

- **Loglikelihood-MC (Tier 1):** the harness never lets the model generate. For
  each answer choice it computes the model's log-probability of that exact answer
  text given the question, and picks the highest. `acc` = fraction correct;
  `acc_norm` = same but length-normalized (divides by answer token count, so long
  answers aren't unfairly penalized). Random-guess floor = 1/(#choices).
  → Measures *latent knowledge / preference*, cheap and deterministic, but the
  model can't "show its work."
- **Generative (Tier 2):** the model actually produces text, which is then parsed
  (final number, or per-instruction checks). → Measures *end-to-end capability*,
  including reasoning and instruction-following — the regime reasoning models are
  trained for.

### Tier 1 — knowledge / commonsense (loglikelihood-MC)

| Benchmark | What it measures | # choices / floor | Why recommended |
|---|---|---|---|
| **arc_challenge** | Grade-school science questions, deliberately the "hard" ARC subset that retrieval/co-occurrence can't solve | 4-way → 0.25 | Standard science-reasoning knowledge probe; in every model card, so directly comparable |
| **hellaswag** | Commonsense sentence completion — pick the plausible continuation of a scene; adversarially filtered so the wrong endings are *almost* plausible | 4-way → 0.25 | Tests commonsense/world-model, not memorized facts; robust, widely reported |
| **winogrande** | Pronoun-resolution (Winograd schemas) — which noun does "it/they" refer to; requires real-world reasoning, not grammar | 2-way → 0.50 | Isolates commonsense coreference; binary so noisy per-item but a clean reasoning signal |
| **mmlu** | 57 subjects (STEM, humanities, law, medicine…) of exam-style knowledge | 4-way → 0.25 | The broad-knowledge yardstick; subject breakdown shows where a model is strong/weak |

**Why this set for the *instruct* models (Llama-3.2-3B, Phi-4-mini):** these are
the canonical "base capability" benchmarks. Standard instruct models answer
directly, so loglikelihood-MC reflects their real knowledge, and the numbers line
up with published model cards — ideal for an apples-to-apples comparison.

**Why it's only a *floor* for reasoning models (R1-Distill, gpt-oss):** see the
"Why two task tiers" section — these models want to think before answering, so
forcing a direct answer collapses the choices toward chance. The score is still
worth recording (it bounds worst-case direct-answer behavior) but is not their
true capability.

### Tier 2 — generative

| Benchmark | What it measures | How it's scored | Why recommended |
|---|---|---|---|
| **gsm8k_cot_zeroshot** | Grade-school math word problems requiring multi-step arithmetic reasoning | Model generates a chain-of-thought, harness extracts the final number and checks exact match | The headline reasoning test: lets R1-Distill / gpt-oss / Qwen3 actually *use* their CoT, which is exactly what they're optimized for. Zero-shot CoT avoids few-shot prompt-format bias |
| **ifeval** | Instruction-following — verifiable constraints (e.g. "answer in 3 bullet points", "include the word X", "no commas") | Programmatic per-instruction pass/fail (no model judge), reported as prompt- and instruction-level accuracy | Measures whether an *instruct-tuned* model obeys formatting/constraints — a capability LL-MC can't see at all. Cheap, objective, discriminates well across all 5 models |

**Why this set for *all* models:** it's the only tier that fairly scores the
reasoning models, and it adds a capability axis (math reasoning + instruction
adherence) that the Tier-1 knowledge probes miss. For the standard instruct
models it's a useful complement; for the reasoning models it's the primary metric.

### Per-model "why" in one line

- **Llama-3.2-3B-Instruct / Phi-4-Mini-Instruct** — Tier 1 is the headline
  (direct-answer models, matches model cards); Tier 2 adds reasoning + instruction
  axes.
- **Qwen3-4B** — hybrid; Tier 2 is the fair headline. Tier 1 is valid only with
  the chat template applied / thinking disabled, else it self-sabotages.
- **DeepSeek-R1-Distill-Qwen-1.5B** — distilled pure reasoner; gsm8k_cot is where
  it shines, Tier 1 is a floor only.
- **GPT-OSS-20B** — harmony reasoning model; Tier 2 is the headline. (And fix the
  corrupt ONNX build first — Tier-1 *and* Tier-2 are meaningless on gibberish.)

---

## Commands (via `scripts/lm_eval_runner.py`)

The runner auto-applies chat templates to `*instruct*` and `*deepseek-r1-distill*`
models in `auto` mode, and (correctly) skips chat-templating pure LL-MC tasks.
Add `*qwen*` for Qwen3 via `--chat-template-model-patterns`.

### Tier 1 — loglikelihood-MC, all models, per backend

```bash
# OpenVINO
python scripts/lm_eval_runner.py --backend openvino \
  --tasks arc_challenge winogrande hellaswag mmlu \
  --batch-size 1 --log-samples

# llama.cpp (managed llama-server, all GPU layers)
python scripts/lm_eval_runner.py --backend llama.cpp \
  --tasks arc_challenge winogrande hellaswag mmlu \
  --batch-size 1 --log-samples

# ONNX Runtime (WebGPU). NOTE: skip GPT-OSS-20B until reconverted (corrupt).
python scripts/lm_eval_runner.py --backend onnxruntime \
  --tasks arc_challenge winogrande hellaswag mmlu \
  --batch-size 1 --log-samples --non-finite-policy error
```

### Tier 2 — generative, all models, per backend

```bash
# OpenVINO
python scripts/lm_eval_runner.py --backend openvino \
  --tasks gsm8k_cot_zeroshot ifeval \
  --batch-size 1 --log-samples --chat-template-mode always

# llama.cpp
python scripts/lm_eval_runner.py --backend llama.cpp \
  --tasks gsm8k_cot_zeroshot ifeval \
  --batch-size 1 --log-samples --chat-template-mode always
```

> Generative tasks **must** be chat-templated for instruct/reasoning models, so
> use `--chat-template-mode always` here (Tier 1 leaves it `auto`).
> ONNX Runtime backend: generation works, but validate the model first.

### Target a single model / single backend

```bash
# one model by directory-name filter
python scripts/lm_eval_runner.py --backend llama.cpp --model "*Qwen3-4B*" \
  --tasks gsm8k_cot_zeroshot ifeval --chat-template-mode always --log-samples

# Qwen3-4B on Tier 1: force chat template so thinking mode doesn't depress LL-MC
python scripts/lm_eval_runner.py --backend openvino --model "*Qwen3-4B*" \
  --tasks arc_challenge winogrande hellaswag mmlu \
  --chat-template-mode always --chat-template-model-patterns "*qwen*" --log-samples
```

### Smoke test before any long run (catches corrupt models)

```bash
# tiny limit to confirm the model + backend produce sane numbers fast
python scripts/lm_eval_runner.py --backend onnxruntime --model "*gpt-oss*" \
  --tasks arc_challenge --limit 20 --log-samples --non-finite-policy warn
```

For ONNX specifically, also run the greedy-coherence check from
[GPT_OSS_LOW_ACCURACY.md](GPT_OSS_LOW_ACCURACY.md) (`scripts/diag_logits.py`) —
if greedy output is gibberish, do not benchmark; reconvert.

---

## Per-model guidance summary

| Model | Tier 1 (LL-MC) | Tier 2 (generative) | Special handling |
|---|---|---|---|
| Llama-3.2-3B-Instruct | ✅ primary | ✅ | chat template auto-applied |
| Phi-4-Mini-Instruct | ✅ primary | ✅ | chat template auto-applied |
| Qwen3-4B | ⚠️ use as floor | ✅ primary | force chat template / disable thinking for Tier 1 |
| DeepSeek-R1-Distill-1.5B | ⚠️ floor only | ✅ primary | chat template auto-applied |
| GPT-OSS-20B | ⚠️ floor only | ✅ primary | **reconvert ONNX first**; OV/GGUF healthy |

## Optional — stronger / leaderboard-comparable tasks

If you later want a harder, more discriminating suite (longer runtimes):

- `mmlu_pro` — 10-way MC, CoT-friendly; better than plain `mmlu` for capable models.
- `gpqa` (`gpqa/cot_zeroshot` or `gpqa/generative`) — graduate-level reasoning.
- `bbh` — Big-Bench Hard reasoning.
- `truthfulqa` — factuality / calibration.

These are not needed for a first comparison pass; the Tier 1 + Tier 2 set above
is the recommended baseline.

---

## Generative / CoT evaluation of multiple-choice tasks

The fair way to score a reasoning model on a multiple-choice question is to let
it **generate** an answer (optionally with chain-of-thought) and then **extract
the chosen letter** from the text — instead of comparing answer log-probs. In
lm-eval this is just a task whose `output_type` is `generate_until` (not
`multiple_choice`), with a `filter_list` that regex-extracts the letter and an
`exact_match` metric. You don't write code — you pick a task variant that already
does this.

### Variants available in this repo

| Task | Style | What it does |
|---|---|---|
| `arc_challenge_chat` | generative MC (direct) | Presents A/B/C/D, prompts *"Your response should end with: The best answer is [letter]"*, generates, extracts the letter, `exact_match`. Drop-in replacement for `arc_challenge`. |
| `leaderboard_mmlu_pro` | MC (10-way) | Harder MMLU; this leaderboard copy is plain `multiple_choice`. For *generative* CoT MMLU use a CoT-prompted variant or `mmlu_pro` with chat template. |
| `gpqa_main_generative_n_shot` (also `_diamond_`, `_extended_`) | generative CoT | Free-form answer ending in *"The answer is (X)"*; two filters — `strict-match` (exact format) and `flexible-extract` (any `(A-D)` in the text) — then `exact_match`. |
| `gpqa_*_cot_zeroshot` / `gpqa_*_cot_n_shot` | generative CoT | Explicit "think step by step" CoT prompt before the letter. |
| `gsm8k_cot_zeroshot` | generative CoT | Already in Tier 2; canonical CoT-extract pattern (extracts the final number). |

### How the extraction works (concrete: `arc_challenge_chat`)

```yaml
output_type: generate_until                 # <- model GENERATES, not log-prob scored
doc_to_text: '... choose the best answer ... Your response should end with
              "The best answer is [the_answer_letter]" ...'
gen_prefix: 'The best answer is'            # primes the model's output
generation_kwargs:
  max_gen_toks: 100                          # <- SEE WARNING BELOW
  until: ["\n\n", "."]
filter_list:                                 # <- pulls the letter out of the text
  - name: remove_whitespace
    filter: [{function: remove_whitespace}, {function: take_first}]
metric_list:
  - metric: exact_match                      # <- compares extracted letter to gold
```

`gpqa_*_generative` uses a stronger extractor — a `flexible-extract` filter that
finds the last `(A)`-style token anywhere in the output — which tolerates a
verbose CoT that wanders before committing.

### Run it

```bash
# Generative MC for arc (fair to reasoning models) — chat template ON
python scripts/lm_eval_runner.py --backend llama.cpp \
  --tasks arc_challenge_chat gpqa_main_generative_n_shot \
  --batch-size 1 --log-samples --chat-template-mode always
```

> ⚠️ **`max_gen_toks` is the #1 gotcha for reasoning models.** `arc_challenge_chat`
> caps generation at **100 tokens**. A reasoning model (R1-Distill, gpt-oss) emits
> a long analysis/thinking block first and may be **truncated before it ever
> writes the letter** → the extractor finds nothing → scored wrong. When running
> reasoning models on these tasks, raise the budget, e.g. append to
> `--extra-model-args` or override the task's `generation_kwargs` to
> `max_gen_toks: 1024+`. Prefer the `gpqa_*_generative`/`*_cot` variants, which
> are built for long CoT, over the Llama-tuned `arc_challenge_chat`.

### Bottom line

- **Reasoning models (R1-Distill, gpt-oss):** prefer generative/CoT variants
  (`*_chat`, `*_generative`, `*_cot_zeroshot`) as the headline, with a generous
  `max_gen_toks`. This is the "let it produce analysis + final answer, then
  extract the letter" path.
- **Standard instruct models:** generative MC works too, but plain
  loglikelihood-MC is already fair to them and cheaper, so keep Tier 1 as their
  headline.

---

## Does chat-templating hurt loglikelihood-MC accuracy?

**It depends on the model — and the honest answer is "it can go either way,"
which is exactly why the runner gates it.**

### What chat-templating does to a loglikelihood-MC request

For a `multiple_choice` task, the harness scores `P(answer | context)`. With
`--apply_chat_template`, the *context* is wrapped in the model's chat format
(e.g. `<|user|>…<|assistant|>`) before the answer continuation is appended and
scored. So you're changing the conditioning prefix, not the scoring math.

### Why it can hurt

- **For base / non-chat models:** wrapping the prompt in chat special tokens the
  model never saw in pretraining shifts the distribution and usually **lowers**
  MC accuracy. Chat templating a base model on arc/hellaswag is counter-productive.
- **Format/answer-boundary effects:** chat templates often append a trailing
  assistant header / whitespace, which changes how the first continuation token
  is scored and can mildly distort `acc`/`acc_norm`. Few-shot MC also gets messier
  (`fewshot_as_multiturn` vs a single block).
- **Published MC leaderboard numbers are *non*-chat-templated.** If you chat-template,
  your arc/mmlu numbers stop being comparable to model cards.

### Why it sometimes helps

- **For instruct/RLHF models**, the chat template is the distribution they were
  tuned on; the matching format can **raise** MC accuracy and is arguably the
  "correct" way to query them.

So "chat template hurts loglikelihood accuracy" is **true for base models and for
matching published numbers, but not universally** — for an instruct model it can
improve it.

### What this repo does (and the safe default)

The runner's `--chat-template-mode auto` (default) **applies chat templates only
to generative tasks and skips pure loglikelihood-MC tasks** — precisely to avoid
the distortion above. That's the safe choice:

- **Tier 1 (loglikelihood-MC): leave `--chat-template-mode auto`** → no chat
  template on arc/hellaswag/winogrande/mmlu → comparable to model cards, no
  format penalty.
- **Tier 2 (generative): use `--chat-template-mode always`** → chat template ON,
  because generation *needs* the instruct format to behave.
- **Exception — Qwen3-4B on Tier 1:** if its thinking mode is on by default,
  bare-prompt MC self-sabotages, so for that one model forcing the chat template
  (or disabling thinking) can help. Test both and report whichever you use.

> Rule of thumb: **don't** chat-template loglikelihood-MC for base models or when
> you need card-comparable numbers; **do** chat-template all generative tasks and
> instruct-model MC where you want "as the model is actually used" numbers — and
> say which you did, because the two aren't comparable.
