"""Gate Rust against Python token IDs, logits, decisions, and probabilities."""
import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--atol", type=float, default=0.0001)
    ap.add_argument("--logit-atol", type=float, default=0.001)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    ref = [json.loads(s) for s in args.reference.read_text().splitlines()]
    cand = [json.loads(s) for s in args.candidate.read_text().splitlines()]
    assert ref and len(ref) == len(cand), (len(ref), len(cand))
    max_logit = max_prob = max_serialized = 0.
    questions = flips = 0
    for a, b in zip(ref, cand):
        assert a["encoding"] == b["encoding"], "tokenization differs"
        assert list(a["logits"]) == list(b["logits"]), "question order differs"
        for q in a["logits"]:
            za, zb = a["logits"][q], b["logits"][q]
            assert list(za) == list(zb), "option order differs"
            max_logit = max(max_logit, *(abs(za[k]-zb[k]) for k in za))
            def softmax(z):
                exp = {k:math.exp(v-max(z.values())) for k,v in z.items()}
                return {k:v/sum(exp.values()) for k,v in exp.items()}
            pa, pb = softmax(za), softmax(zb)
            max_prob = max(max_prob, *(abs(pa[k]-pb[k]) for k in pa))
            flips += max(za, key=za.get) != max(zb, key=zb.get)
            aa, ab = a["answers"][q], b["answers"][q]
            if aa["type"] == "noul":
                max_serialized = max(max_serialized, abs(aa["noul"]-ab["noul"]))
            else:
                max_serialized = max(max_serialized, *(abs(p-ab["probabilities"][k]) for k,p in aa["probabilities"].items()))
                assert abs(aa["confidence"]-ab["confidence"]) <= args.atol + 1e-8
                if aa["type"] == "score":
                    assert abs(aa["score"]-ab["score"]) < args.atol * len(za)
                    assert aa["legend"] == ab["legend"]
            questions += 1
    ra = statistics.median(a["latency_ms"] for a in ref)
    rb = statistics.median(b["latency_ms"] for b in cand)
    report = {"records":len(ref),"questions":questions,"max_logit_error":max_logit,
              "max_probability_error":max_prob,"argmax_flips":flips,"token_ids_identical":True,
              "max_serialized_probability_error":max_serialized,
              "reference_median_ms":ra,"candidate_median_ms":rb,"speedup":ra/rb,
              "atol":args.atol,"logit_atol":args.logit_atol,
              "passed":max_prob < args.atol and max_logit < args.logit_atol and flips == 0}
    if args.out: args.out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    assert report["passed"], "parity gate failed"


if __name__ == "__main__":
    main()
