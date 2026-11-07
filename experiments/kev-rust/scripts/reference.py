"""Measure the official fp32 unmerged Python path on our Rust parity fixtures."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "kev"))
from kev.api import SystemOneRequest, to_record, to_answers
from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import rows_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "weights")
    ap.add_argument("--input", type=Path, default=ROOT / "fixtures" / "requests.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "python.jsonl")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(2)
    meta = json.loads((args.weights / "metadata.json").read_text())
    start = time.perf_counter()
    tok, model = Checkpoint(meta["kev_path"]).load("cpu", LoadOptions(merge=False, attn="eager", dtype=torch.float32))
    print(f"Loaded reference in {time.perf_counter()-start:.2f}s", file=sys.stderr, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f, torch.inference_mode():
        for line in args.input.read_text().splitlines():
            req = SystemOneRequest.model_validate(json.loads(line))
            rec, questions = to_record(req)
            enc = model.encode(tok, rec, max_state=65536, max_branch=73728, strict=True)
            state, _, branches = rows_of(enc)
            for iteration in range(args.repeats):
                start = time.perf_counter()
                zs = model(enc)
                latency = (time.perf_counter()-start)*1000
                ps = [torch.softmax(z, -1).tolist() for z in zs]
                out = {"answers": to_answers(ps, questions), "logits": {q["id"]: dict(zip(q["keys"], z.tolist())) for q,z in zip(questions,zs)},
                       "latency_ms":latency, "iteration":iteration,
                       "encoding":{"state":state,"branches":{q["id"]:{"ids":br["ids"],"ends":br["opts"]} for q,br in zip(questions,branches)}}}
                f.write(json.dumps(out)+"\n"); f.flush()
                print(f"record={len(state)} state tokens, {len(zs)} questions: {latency:.0f}ms", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
