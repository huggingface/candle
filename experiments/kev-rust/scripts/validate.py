"""Run native tests, Python parity, Rust ablations, and warmed CPU benchmarks sequentially."""
import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, output=None):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    if output:
        with output.open("w") as f:
            subprocess.run(list(map(str,cmd)), cwd=ROOT, env={**os.environ,"RAYON_NUM_THREADS":"2","OMP_NUM_THREADS":"2"}, stdout=f, check=True)
    else:
        subprocess.run(list(map(str,cmd)), cwd=ROOT, env={**os.environ,"RAYON_NUM_THREADS":"2","OMP_NUM_THREADS":"2"}, check=True)


def read(p):
    return [json.loads(s) for s in p.read_text().splitlines()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--reuse-python-parity", action="store_true", help="Reuse the existing reference for unchanged checkpoint, source and fixtures")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    assert args.repeats >= 2
    result = ROOT / "results"
    result.mkdir(exist_ok=True)
    if not args.skip_build:
        run(["cargo","test","--release"])
        run(["cargo","build","--release"])
    binary = ROOT / "target/release/kev-rust"
    requests = ROOT / "fixtures/parity-v2.jsonl"
    # Correctness first: compare unmerged official Python against three independent native modes.
    if not args.reuse_python_parity:
        run([sys.executable, ROOT/"scripts/reference.py", "--input",requests,"--out",result/"python-parity.jsonl"])
    parity = {}
    for name, flags in [("rows-tensor",["--rows","--tensor"]),("rows-fused",["--rows","--unbatched"]),("prefix-fused",["--unbatched"]),("prefix-batched",[])]:
        run([binary,"--input",requests,*flags],result/f"{name}-parity.jsonl")
        run([sys.executable,ROOT/"scripts/compare.py",result/"python-parity.jsonl",result/f"{name}-parity.jsonl","--out",result/f"{name}-parity.json"])
        parity[name]=json.loads((result/f"{name}-parity.json").read_text())
    native=read(result/"prefix-batched-parity.jsonl")
    assert native[-1]["state_tokens"] > 256, "long fixture must cross a prefill boundary"
    for q, z in native[0]["logits"].items():
        assert max(abs(v-native[4]["logits"][q][k]) for k,v in z.items()) < 1e-5, "question reorder changes another question"
    for i,q in enumerate(native[0]["logits"],start=5):
        assert max(abs(v-native[i]["logits"][q][k]) for k,v in native[0]["logits"][q].items()) < 1e-5, "question isolation failed"
    # Performance is a separate sequential run after builds and downloads have finished.
    bench=ROOT/"fixtures/requests.jsonl"
    run([sys.executable,ROOT/"scripts/reference.py","--input",bench,"--out",result/"python-bench.jsonl","--repeats",args.repeats])
    benchmarks={}
    for name, flags in [("rows-tensor",["--rows","--tensor"]),("rows-fused",["--rows","--unbatched"]),("prefix-fused",["--unbatched"]),("prefix-batched",[])]:
        run([binary,"--input",bench,"--repeats",args.repeats,*flags],result/f"{name}-bench.jsonl")
        run([sys.executable,ROOT/"scripts/compare.py",result/"python-bench.jsonl",result/f"{name}-bench.jsonl","--out",result/f"{name}-bench.json"])
    for name in ["python","rows-tensor","rows-fused","prefix-fused","prefix-batched"]:
        rows=read(result/f"{name}-bench.jsonl")
        warmed=[r["latency_ms"] for r in rows if r["iteration"]>0]
        benchmarks[name]={"warm_median_ms":statistics.median(warmed),"samples":len(warmed),
                          "by_record_ms":[statistics.median(r["latency_ms"] for r in rows[i*args.repeats:(i+1)*args.repeats] if r["iteration"]>0) for i in range(len(rows)//args.repeats)]}
    optimized=benchmarks["prefix-batched"]["warm_median_ms"]
    report={"hardware":{"architecture":platform.machine(),"cpu_count":os.cpu_count(),"threads":2},
            "packages":{k:importlib.metadata.version(k) for k in ["torch","transformers","peft","tokenizers","safetensors"]},
            "candle_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT).decode().strip(),
            "kev_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT.parent/"kev").decode().strip(),
            "checkpoint":json.loads((ROOT/"weights/metadata.json").read_text()),"parity":parity,
            "isolation_passed":True,"chunk_boundary_passed":True,"benchmarks":benchmarks,
            "speedup_vs_python":benchmarks["python"]["warm_median_ms"]/optimized,
            "speedup_vs_native_baseline":benchmarks["rows-tensor"]["warm_median_ms"]/optimized}
    (result/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
