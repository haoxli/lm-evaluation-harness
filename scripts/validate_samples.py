#!/usr/bin/env python
"""Validate lm-eval samples: for each scored doc decide whether a score of 0
is a genuine MODEL-accuracy issue (a well-formed answer was returned and
scored, it's just wrong) or a FRAMEWORK/DATA issue (the model's answer was not
captured/scored correctly: empty output, extraction returned nothing,
generation truncated before any answer, or the gold label itself is wrong).

Benign generation artifacts (e.g. a correct answer followed by a repetition
loop) are NOT counted as invalid, because the answer WAS returned and scored.

Usage: python validate_samples.py <result_dir>
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter


def load_samples(path):
    txt = open(path, encoding="utf-8").read().strip()
    if not txt:
        return
    if txt[0] == "[":
        for o in json.loads(txt):
            yield o
        return
    dec = json.JSONDecoder()
    i, n = 0, len(txt)
    while i < n:
        while i < n and txt[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        obj, e = dec.raw_decode(txt, i)
        yield obj
        i = e


NUM = re.compile(r"-?\d[\d,]*\.?\d*")
LETTER = re.compile(r"\b([A-E])\b")


def get_resp(o):
    try:
        return o["resps"][0][0]
    except Exception:
        return ""


def get_filt(o):
    fr = o.get("filtered_resps") or []
    return str(fr[0]) if fr else ""


def bad_gold_gsm8k(o):
    """Flag docs whose #### gold is produced by an arithmetically wrong final calc."""
    doc = o.get("doc", {})
    ans = doc.get("answer")
    target = o.get("target")
    if not ans or target is None:
        return None
    calcs = re.findall(r"<<([^=]+)=([^>]+)>>", ans)
    if not calcs:
        return None
    expr, res = calcs[-1]
    try:
        res_v = float(res.replace(",", ""))
        tgt_v = float(str(target).replace(",", ""))
    except ValueError:
        return None
    if abs(res_v - tgt_v) > 1e-6:
        return None  # last calc isn't the final answer; skip
    try:
        computed = eval(expr.replace("x", "*").replace(",", "").replace(" ", ""))  # noqa: S307
    except Exception:
        return None
    if abs(computed - res_v) > 1e-6:
        return f"dataset final calc {expr.strip()}={res} is wrong (should be {computed:g}); gold {target} is corrupt"
    return None


def analyze_gsm8k(path):
    valid_model, framework, data = [], [], []
    correct = 0
    rep_but_ok = 0
    for o in load_samples(path):
        # only judge on the primary flexible-extract filter (strict-match is a
        # stricter secondary metric that legitimately voids non-"answer is" text)
        if o.get("filter") and o.get("filter") != "flexible-extract":
            continue
        did = o.get("doc_id")
        resp = get_resp(o)
        filt = get_filt(o)
        score = o.get("exact_match", 0.0)
        if score == 1.0:
            correct += 1
        has_num = bool(NUM.search(filt)) and filt.strip().lower() != "[invalid]"
        reason = None
        if not resp.strip():
            reason = ("framework", "EMPTY generation")
        elif not has_num:
            if re.search(r"answer is\s*\$?-?\d", resp, re.I) or re.search(r"####", resp):
                reason = ("framework", f"extraction MISSED a stated answer (filtered={filt!r})")
            elif not re.search(r"(answer is|####|=\s*\$?\d)", resp, re.I):
                reason = ("framework", "no answer produced (truncated/rambling)")
            else:
                reason = ("framework", f"extraction failed (filtered={filt!r})")
        else:
            g = bad_gold_gsm8k(o)
            if g and score == 0.0:
                reason = ("data", g)
        if reason is None:
            if re.search(r"(.{5,80}?)\1{4,}", resp):
                rep_but_ok += 1
            if score == 0.0:
                valid_model.append(did)
        elif reason[0] == "framework":
            framework.append((did, reason[1]))
        else:
            data.append((did, reason[1]))
    return dict(correct=correct, valid_model=valid_model,
                framework=framework, data=data, rep_but_ok=rep_but_ok)


def analyze_arc(path):
    framework = []
    correct = 0
    valid_model = []
    total = 0
    for o in load_samples(path):
        total += 1
        did = o.get("doc_id")
        resp = get_resp(o)
        filt = get_filt(o)
        score = o.get("exact_match", o.get("acc", 0.0))
        if score == 1.0:
            correct += 1
        if not resp.strip():
            framework.append((did, "EMPTY generation"))
        elif not LETTER.search(filt) and not LETTER.search(resp[:20]):
            framework.append((did, f"no A-E letter (filtered={filt!r}, resp={resp[:20]!r})"))
        elif score == 0.0:
            valid_model.append(did)
    return dict(total=total, correct=correct, valid_model=valid_model, framework=framework)


def main(result_dir):
    files = sorted(glob.glob(os.path.join(result_dir, "samples_*")))
    print("=" * 72)
    print("GENERATIVE TASKS — validity of returned answers")
    print("=" * 72)

    gsm = [f for f in files if os.path.basename(f).startswith("samples_gsm8k")]
    arc = [f for f in files if os.path.basename(f).startswith("samples_arc_challenge_chat")]

    for f in gsm:
        r = analyze_gsm8k(f)
        n = len(r["valid_model"]) + len(r["framework"]) + len(r["data"]) + r["correct"]
        print(f"\n[gsm8k_cot]  docs≈{n}  correct={r['correct']}")
        print(f"  VALID   (model answered; wrong => model-accuracy)      : {len(r['valid_model'])}")
        print(f"  INVALID (framework failed to capture/produce answer)   : {len(r['framework'])}")
        print(f"  INVALID (bad GOLD label in dataset)                    : {len(r['data'])}")
        print(f"  (note) benign correct-then-repetition-loop answers     : {r['rep_but_ok']}")
        fc = Counter(reason.split("(")[0].strip() for _, reason in r["framework"])
        for reason, c in fc.most_common():
            print(f"      {c:4d}  framework: {reason}")
        if r["framework"]:
            print("      framework doc_ids:", ", ".join(str(d) for d, _ in r["framework"][:60]),
                  "..." if len(r["framework"]) > 60 else "")
        if r["data"]:
            print("      bad-gold doc_ids:", ", ".join(str(d) for d, _ in r["data"][:60]),
                  "..." if len(r["data"]) > 60 else "")
            print("      example:", r["data"][0])

    for f in arc:
        r = analyze_arc(f)
        print(f"\n[arc_challenge_chat]  docs={r['total']}  correct={r['correct']}")
        print(f"  VALID   (letter returned; wrong => model-accuracy)     : {len(r['valid_model'])}")
        print(f"  INVALID (no valid letter / empty)                      : {len(r['framework'])}")
        if r["framework"]:
            print("      invalid doc_ids:", ", ".join(str(d) for d, _ in r["framework"][:60]),
                  "..." if len(r["framework"]) > 60 else "")
            print("      examples:", r["framework"][:5])

    mc = [f for f in files if os.path.basename(f).startswith("samples_mmlu")]
    print("\n" + "=" * 72)
    print(f"MULTIPLE-CHOICE (mmlu) — {len(mc)} subtasks (loglikelihood)")
    print("=" * 72)
    bad_total = 0
    for f in mc:
        bad = []
        for o in load_samples(f):
            if not o.get("resps") or not any(o["resps"]):
                bad.append(o.get("doc_id"))
        if bad:
            bad_total += len(bad)
            print(f"  {os.path.basename(f)}: {len(bad)} invalid -> {bad[:10]}")
    if bad_total == 0:
        print("  All subtasks valid: every doc produced loglikelihood responses "
              "(argmax scoring always well-defined; wrong picks are model-accuracy).")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else r"C:\workspace\project\llm\lm-evaluation-harness\results\20260708_173338\onnxruntime\Aion-1.0-Instruct"
    main(d)
