#!/usr/bin/env python
"""
Minimal reproducer: ONNX Runtime GenAI WebGPU returns all-NaN logits.

This is a self-contained bug report. It depends ONLY on:
    pip install numpy onnxruntime-genai onnxruntime-webgpu
(no lm-evaluation-harness, no torch, no datasets).

What it does
------------
For each of a handful of multiple-choice questions, it scores every answer with
teacher-forced log-likelihood -- exactly the access pattern a benchmark harness
uses:

    gen = og.Generator(model, params)
    gen.append_tokens(context_tokens)        # prompt
    for tok in continuation_tokens:          # gold answer tokens
        logits = gen.get_logits()            # next-token logits
        gen.append_tokens([tok])             # teacher forcing

On the WebGPU execution provider with the DeepSeek-R1-Distill-Qwen-1.5B ONNX
model, `gen.get_logits()` intermittently returns a logits vector that is
ENTIRELY non-finite (all 151,936 values NaN). The same request scored on the
CPU EP, or with WebGPU graph capture disabled, returns finite logits.

Observed properties (strong signal that this is a WebGPU graph-capture /
buffer-aliasing bug, not bad input):
  * Within ONE question, some answer choices score finite and others NaN,
    even though they share the same prompt.
  * Whether a given (prompt, answer) pair goes NaN depends on GPU/session
    state, not on the text: scored cold it may be NaN; scored after other
    requests have warmed the model it may be finite.
  * genai_config.json for this model sets:
        "provider_options": [{"webgpu": {"enableGraphCapture": "1"}}]
        "past_present_share_buffer": true

Usage
-----
    python repro_webgpu_nan.py --model <path-to-onnx-model-dir>

    # confirm graph capture is the trigger (scores via a temp config copy that
    # drops enableGraphCapture):
    python repro_webgpu_nan.py --model <dir> --no-graph-capture

By default, the script always exits 0 so batch runs can continue even if NaN is
observed. Use --fail-on-nan for strict CI-style failure signaling.

The embedded questions (webgpu_nan_questions.json, sitting next to this script)
were captured from a real run; `nan_flags` records which choices produced NaN
in that run so you can see expected-vs-actual at a glance.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime_genai as og


HERE = Path(__file__).resolve().parent
QUESTIONS_FILE = HERE / "webgpu_nan_questions.json"


def load_model(model_dir: Path, disable_graph_capture: bool) -> "og.Model":
    """Load the model. Optionally rewrite genai_config.json (in a temp copy) to
    drop enableGraphCapture, without touching the user's real model directory."""
    effective_dir = model_dir
    if disable_graph_capture:
        tmp = Path(tempfile.mkdtemp(prefix="repro_nan_nocap_"))
        for f in model_dir.iterdir():
            if f.name == "genai_config.json":
                continue
            dst = tmp / f.name
            try:
                dst.symlink_to(f)
            except (OSError, NotImplementedError):
                shutil.copy2(f, dst)
        cfg = json.loads((model_dir / "genai_config.json").read_text("utf-8"))
        provs = (
            cfg.get("model", {})
            .get("decoder", {})
            .get("session_options", {})
            .get("provider_options", [])
        )
        for p in provs:
            if isinstance(p, dict) and "webgpu" in p:
                p["webgpu"].pop("enableGraphCapture", None)
        (tmp / "genai_config.json").write_text(json.dumps(cfg, indent=2), "utf-8")
        effective_dir = tmp
        print(f"[setup] enableGraphCapture DISABLED via temp config: {tmp}")

    print(f"[setup] loading model from {effective_dir}")
    return og.Model(str(effective_dir))


def score_choice(model, tokenizer, context: str, continuation: str) -> tuple[float, bool]:
    """Teacher-forced log-likelihood of `continuation` given `context`.

    Returns (sum_log_prob, saw_non_finite). Mirrors the harness access pattern:
    append context once, then for each gold token read get_logits() and append it.
    """
    full = list(tokenizer.encode(context + continuation))
    ctx_tokens = list(tokenizer.encode(context))
    cont_tokens = full[len(ctx_tokens):]
    if not cont_tokens:
        return 0.0, False

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=4096, do_sample=False)
    gen = og.Generator(model, params)
    gen.append_tokens(np.array(full[: len(ctx_tokens)], dtype=np.int32))

    total = 0.0
    for tok in cont_tokens:
        logits = np.array(gen.get_logits(), dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(logits)):
            return float("nan"), True
        mx = float(np.max(logits))
        lse = mx + float(np.log(np.sum(np.exp(logits - mx))))
        total += float(logits[int(tok)]) - lse
        gen.append_tokens(np.array([int(tok)], dtype=np.int32))
    return total, False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce WebGPU all-NaN logits in onnxruntime-genai",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the onnxruntime-genai model directory (contains genai_config.json)",
    )
    parser.add_argument(
        "--no-graph-capture",
        action="store_true",
        help="Score with enableGraphCapture removed (expected: no NaN)",
    )
    parser.add_argument(
        "--questions",
        default=str(QUESTIONS_FILE),
        help="Path to the embedded questions JSON",
    )
    parser.add_argument(
        "--fail-on-nan",
        action="store_true",
        help="Exit non-zero when NaN is observed (default: always exit 0)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model)
    if not (model_dir / "genai_config.json").exists():
        print(f"ERROR: no genai_config.json under {model_dir}", file=sys.stderr)
        return 2

    questions = json.loads(Path(args.questions).read_text("utf-8"))

    import onnxruntime as ort

    print(f"[env] onnxruntime-genai {og.__version__}")
    print(f"[env] onnxruntime providers: {ort.get_available_providers()}")

    model = load_model(model_dir, args.no_graph_capture)
    tokenizer = og.Tokenizer(model)

    total_choices = 0
    nan_choices = 0
    for qi, q in enumerate(questions):
        ctx = q["context"]
        prompt_line = ctx.replace("\n", " ").strip()
        print(f"\nQ{qi}: {prompt_line[:90]}", flush=True)
        for ci, choice in enumerate(q["choices"]):
            total_choices += 1
            print(f"    scoring choice {ci}...", flush=True)
            ll = float("nan")
            nan = False
            err = None
            try:
                ll, nan = score_choice(model, tokenizer, ctx, choice)
                if nan:
                    nan_choices += 1
            except Exception as ex:
                # Keep the run going so every question still gets reported.
                err = f"{type(ex).__name__}: {ex}"
                nan = True
                nan_choices += 1
            expected = ""
            flags = q.get("nan_flags")
            if flags and ci < len(flags):
                expected = "  (recorded: NaN)" if flags[ci] else "  (recorded: ok)"
            status = "NaN " if nan else f"{ll:8.3f}"
            if err:
                print(
                    f"    choice {ci}: {status}  {choice!r}{expected}  [error: {err}]",
                    flush=True,
                )
            else:
                print(f"    choice {ci}: {status}  {choice!r}{expected}", flush=True)

    print("\n=== Summary ===")
    print(f"  choices scored : {total_choices}")
    print(f"  NaN choices    : {nan_choices}")
    if nan_choices and not args.no_graph_capture:
        print("  -> BUG REPRODUCED: WebGPU returned all-NaN logits.")
    elif nan_choices == 0 and args.no_graph_capture:
        print("  -> No NaN with graph capture disabled (confirms the trigger).")
    elif nan_choices == 0:
        print("  -> No NaN this run (the bug is GPU-state-dependent; try re-running).")

    if args.fail_on_nan and nan_choices and not args.no_graph_capture:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
