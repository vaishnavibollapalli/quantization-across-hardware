#!/usr/bin/env python3
"""
Score model outputs on the held-out quality set.

Input: a JSONL file of outputs, one line per (prompt, model, quant):
  {"id": "Q01", "model": "qwen2.5-3b", "quant": "Q4_K_M", "device": "mac-m2", "output": "..."}

Usage:
  python3 score_quality.py outputs.jsonl                 # writes scores.csv + prints summary
  python3 score_quality.py outputs.jsonl --manual manual.csv
      manual.csv columns: id,model,quant,score   (1-5 rubric scores for the open-ended items)

Auto-checked items score 1 (pass) or 0 (fail). Manual items are left blank unless --manual
is given; their 1-5 score is reported separately (never mixed into auto accuracy).
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def strip_fences(text):
    m = re.search(r"```(?:\w+)?\s*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def extract_number(text):
    m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?\d[\d,]*\.?\d*)", text, re.I)
    if not m:
        m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def check(item, output):
    c = item["check"]
    t = c["type"]
    out = output.strip()
    if t == "number":
        got = extract_number(out)
        return int(got is not None and abs(got - c["value"]) <= c["tol"]), f"got={got}"
    if t == "yesno":
        m = re.search(r"\b(yes|no)\b", out.lower())
        got = m.group(1) if m else None
        return int(got == c["value"]), f"got={got}"
    if t == "contains_all":
        flags = 0 if c.get("case_sensitive") else re.I
        missing = [v for v in c["values"] if not re.search(rf"\b{re.escape(v)}\b", out, flags)]
        return int(not missing), f"missing={missing}" if missing else "ok"
    if t == "regex_all":
        missing = [p for p in c["patterns"] if not re.search(p, out, re.I)]
        return int(not missing), f"missing={missing}" if missing else "ok"
    if t == "bullets_exact":
        lines = [l for l in out.splitlines() if l.strip()]
        bullets = [l for l in lines if re.match(r"^\s*-\s+\S", l)]
        ok = len(bullets) == c["n"] and len(lines) == c["n"]
        return int(ok), f"lines={len(lines)} bullets={len(bullets)}"
    if t == "json_keys":
        body = strip_fences(out)
        s, e = body.find("{"), body.rfind("}")
        try:
            obj = json.loads(body[s:e + 1])
            ok = isinstance(obj, dict) and set(obj) == set(c["keys"])
            return int(ok), f"keys={sorted(obj)}" if isinstance(obj, dict) else "not an object"
        except Exception as ex:  # invalid JSON is a fail
            return 0, f"invalid json: {ex.__class__.__name__}"
    if t == "one_sentence_max_words":
        words = len(out.split())
        sentences = len(re.findall(r"[.!?](\s|$)", out)) or 1
        return int(words <= c["n"] and sentences == 1), f"words={words} sentences={sentences}"
    if t == "label":
        m = re.search(r"[a-zA-Z]+", out)
        got = m.group(0).lower() if m else None
        return int(got == c["value"]), f"got={got}"
    if t == "python_tests":
        code = strip_fences(out) + "\n\n" + "\n".join(c["tests"]) + "\n"
        try:
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)
            return int(r.returncode == 0), "pass" if r.returncode == 0 else r.stderr.strip().splitlines()[-1][:120]
        except subprocess.TimeoutExpired:
            return 0, "timeout"
    if t == "contains_any":
        hit = [v for v in c["values"] if re.search(rf"\b{re.escape(v)}\b", out, re.I)]
        return int(bool(hit)), f"hit={hit}"
    if t == "answer_text":
        m = re.findall(r"Answer:\s*\**\s*([A-Za-z0-9]+)", out)
        got = m[-1] if m else None
        return int(got is not None and got.lower() == c["value"].lower()), f"got={got}"
    if t == "exact":
        got = out.strip().strip(".!*\"'` ")
        return int(got == c["value"]), f"got={got[:40]!r}"
    if t == "numbered_exact":
        lines = [l for l in out.splitlines() if l.strip()]
        numbered = [l for l in lines if re.match(r"^\s*\d+[.)]\s+\S", l)]
        ok = len(numbered) == c["n"] and len(lines) == c["n"]
        return int(ok), f"lines={len(lines)} numbered={len(numbered)}"
    if t == "starts_with":
        return int(out.lstrip("*\"' ").startswith(c["value"])), f"start={out[:20]!r}"
    if t == "line_count":
        lines = [l for l in out.splitlines() if l.strip()]
        return int(len(lines) == c["n"]), f"lines={len(lines)}"
    if t == "json_set":
        body = strip_fences(out)
        s, e = body.find("["), body.rfind("]")
        try:
            arr = json.loads(body[s:e + 1])
            ok = isinstance(arr, list) and sorted(arr) == sorted(c["values"])
            return int(ok), f"got={arr}"
        except Exception as ex:
            return 0, f"invalid json: {ex.__class__.__name__}"
    if t == "all_lower":
        letters_ok = not re.search(r"[A-Z]", out)
        missing = [v for v in c["values"] if v not in out]
        return int(letters_ok and not missing), f"lowercase={letters_ok} missing={missing}"
    if t == "prefixed_lines":
        lines = [l for l in out.splitlines() if l.strip()]
        pref = [l for l in lines if l.strip().lstrip("-* ").startswith(c["prefix"])]
        ok = len(pref) == c["n"] and len(lines) == c["n"]
        return int(ok), f"lines={len(lines)} prefixed={len(pref)}"
    if t == "sentence_count":
        n = len(re.findall(r"[.!?]+(\s|$)", out))
        return int(n == c["n"]), f"sentences={n}"
    if t == "manual":
        return None, "manual rubric"
    raise ValueError(f"unknown check type {t}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outputs")
    ap.add_argument("--prompts", default=str(ROOT / "quality_prompts.jsonl"))
    ap.add_argument("--manual", help="CSV with id,model,quant,score for manual items")
    ap.add_argument("--out", default="scores.csv")
    args = ap.parse_args()

    items = {}
    with open(args.prompts, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                it = json.loads(line)
                items[it["id"]] = it

    manual = {}
    if args.manual:
        with open(args.manual, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                manual[(r["id"], r["model"], r["quant"])] = float(r["score"])

    rows = []
    with open(args.outputs, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            it = items.get(o["id"])
            if it is None:
                print(f"skip unknown id {o['id']}", file=sys.stderr)
                continue
            score, detail = check(it, o.get("output", ""))
            if score is None and (o["id"], o["model"], o["quant"]) in manual:
                score = manual[(o["id"], o["model"], o["quant"])]
            rows.append({"id": o["id"], "model": o["model"], "quant": o["quant"],
                         "device": o.get("device", ""), "category": it["category"],
                         "check": it["check"]["type"], "score": "" if score is None else score,
                         "detail": detail})

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)

    auto, man = defaultdict(list), defaultdict(list)
    for r in rows:
        key = (r["model"], r["quant"])
        if r["check"] == "manual":
            if r["score"] != "":
                man[key].append(float(r["score"]))
        else:
            auto[key].append(r["score"])

    print(f"{'model':<22}{'quant':<10}{'auto acc':>10}{'n':>5}{'manual avg (1-5)':>18}")
    for key in sorted(set(auto) | set(man)):
        a = auto.get(key, [])
        m = man.get(key, [])
        acc = f"{sum(a) / len(a):.2f}" if a else "-"
        mavg = f"{sum(m) / len(m):.2f}" if m else "-"
        print(f"{key[0]:<22}{key[1]:<10}{acc:>10}{len(a):>5}{mavg:>18}")
    print(f"\nper-item results -> {args.out}")


if __name__ == "__main__":
    main()
