#!/usr/bin/env python
"""
Diagnostic: compare teacher-forced log-likelihood scoring of arc_challenge-style
choices on WebGPU vs CPU, and inspect raw logit statistics.

Goal: determine WHY arc_challenge accuracy is ~chance (0.227). The recorded
per-token logprobs (~ -12.2 == ln(1/201088)) suggest near-uniform logits.
This script dumps logit stats (max-min spread, top tokens, argmax) so we can
tell whether the logits are genuinely flat (model/EP bug) or discriminative
(harness scoring bug).
"""

from __future__ import annotations

import argparse
import numpy as np
import onnxruntime_genai as og


QUESTION = (
    "Question: An astronomer observes that a planet rotates faster after a "
    "meteorite impact. Which is the most likely effect of this increase in "
    "rotation?\nAnswer:"
)
CHOICES = [
    " Planetary density will decrease.",
    " Planetary years will become longer.",
    " Planetary days will become shorter.",  # gold (target=2)
    " Planetary gravity will become stronger.",
]
GOLD = 2


def build_model(model_dir: str, provider: str) -> og.Model:
    cfg = og.Config(model_dir)
    cfg.clear_providers()
    if provider != "cpu":
        cfg.append_provider(provider)
    return og.Model(cfg)


def logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))


def score(model, tok, context, continuation, dump=False):
    ctx_ids = tok.encode(context)
    full_ids = tok.encode(context + continuation)
    cont_ids = full_ids[len(ctx_ids):]

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=4096, do_sample=False)
    gen = og.Generator(model, params)
    gen.append_tokens(np.array(full_ids[: len(ctx_ids)], dtype=np.int32))

    total = 0.0
    for j, t in enumerate(cont_ids):
        logits = np.array(gen.get_logits(), dtype=np.float32).reshape(-1)
        finite = np.isfinite(logits).all()
        lse = logsumexp(logits)
        lp = float(logits[int(t)] - lse)
        total += lp
        if dump:
            srt = np.argsort(logits)[::-1][:5]
            print(
                f"    tok{j} id={int(t)} lp={lp:.3f} finite={finite} "
                f"min={logits.min():.2f} max={logits.max():.2f} "
                f"spread={logits.max()-logits.min():.2f} "
                f"argmax={int(np.argmax(logits))} top5={srt.tolist()}"
            )
        gen.append_tokens(np.array([int(t)], dtype=np.int32))
    return total, len(cont_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", default="webgpu", help="webgpu|cpu|cuda")
    ap.add_argument("--dump", action="store_true", help="dump per-token logit stats")
    args = ap.parse_args()

    print(f"[env] genai {og.__version__}  provider={args.provider}")
    model = build_model(args.model, args.provider)
    tok = og.Tokenizer(model)

    print(f"\nQ: {QUESTION}")
    lls = []
    for i, ch in enumerate(CHOICES):
        ll, n = score(model, tok, QUESTION, ch, dump=args.dump)
        lls.append(ll)
        flag = " <-- GOLD" if i == GOLD else ""
        print(f"  choice {i}: ll={ll:8.3f}  ll/tok={ll/n:7.3f}  ({n} tok) {ch!r}{flag}")

    pred = int(np.argmax(lls))
    print(f"\n  argmax choice = {pred}  gold = {GOLD}  {'CORRECT' if pred==GOLD else 'WRONG'}")


if __name__ == "__main__":
    main()
