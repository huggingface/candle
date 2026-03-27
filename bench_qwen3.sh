#!/usr/bin/env bash
# Benchmark Qwen3-0.6B: Standard CPU vs CPU Flash Attention
#
# Measures token/s throughput and peak RSS (memory) for:
#   1. standard  — matmul attention (no --use-flash-attn)
#   2. cpu_flash — fused CPU flash attention (--use-flash-attn)
#
# Both use the current working tree build.
#
# Usage:
#   ./bench_qwen3.sh [sample_len]   # default: 50
#
# Requirements:
#   - macOS (uses /usr/bin/time -l for peak RSS)
#   - Model weights auto-downloaded on first run (~1.2 GB for Qwen3-0.6B)

set -euo pipefail

SAMPLE_LEN="${1:-50}"
PROMPT="Explain the theory of relativity in simple terms."
MODEL="3-0.6b"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/bench_results"
BINARY="$SCRIPT_DIR/target/release/examples/qwen"

mkdir -p "$RESULTS_DIR"

# ── Run a single benchmark ───────────────────────────────────────────────
run_bench() {
    local name="$1"
    local use_flash="$2"
    local result_file="$RESULTS_DIR/${name}.txt"

    local flash_flag=""
    if [[ "$use_flash" == "true" ]]; then
        flash_flag="--use-flash-attn"
    fi

    echo "  Running $name (sample_len=$SAMPLE_LEN) ..."

    /usr/bin/time -l "$BINARY" \
        --cpu \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --sample-len "$SAMPLE_LEN" \
        $flash_flag \
        2>"$RESULTS_DIR/${name}_time.txt" \
        >"$RESULTS_DIR/${name}_stdout.txt" \
        || true

    # Extract token/s
    local tokens_per_sec
    tokens_per_sec=$(grep -oE '[0-9]+\.[0-9]+ token/s' "$RESULTS_DIR/${name}_stdout.txt" \
        | tail -1 | grep -oE '[0-9]+\.[0-9]+') || tokens_per_sec="N/A"

    # Extract peak RSS (macOS reports bytes)
    local peak_rss_bytes
    peak_rss_bytes=$(grep "maximum resident set size" "$RESULTS_DIR/${name}_time.txt" \
        | grep -oE '[0-9]+') || peak_rss_bytes="0"
    local peak_rss_mb
    if [[ "$peak_rss_bytes" -gt 0 ]]; then
        peak_rss_mb=$(echo "scale=1; $peak_rss_bytes / 1048576" | bc)
    else
        peak_rss_mb="N/A"
    fi

    # Extract wall time
    local wall_time
    wall_time=$(grep "real" "$RESULTS_DIR/${name}_time.txt" \
        | head -1 | awk '{print $1}') || wall_time="N/A"

    echo "$tokens_per_sec $peak_rss_mb $wall_time" > "$result_file"
}

# ── Main ─────────────────────────────────────────────────────────────────
echo "=== Qwen3-0.6B CPU Attention Benchmark ==="
echo ""
echo "Config: sample_len=$SAMPLE_LEN, model=$MODEL"
echo "Commit: $(git rev-parse --short HEAD)"
echo ""

echo "Step 1: Building release binary"
cargo build --release --example qwen -p candle-examples 2>&1 | tail -3
echo ""

echo "Step 2: Running benchmarks"
run_bench "standard"  "false"
run_bench "cpu_flash" "true"
echo ""

echo "Step 3: Results"
echo ""

# Read results
std_tps="N/A"; std_rss="N/A"; std_wall="N/A"
flash_tps="N/A"; flash_rss="N/A"; flash_wall="N/A"

if [[ -f "$RESULTS_DIR/standard.txt" ]]; then
    read -r std_tps std_rss std_wall < "$RESULTS_DIR/standard.txt"
fi
if [[ -f "$RESULTS_DIR/cpu_flash.txt" ]]; then
    read -r flash_tps flash_rss flash_wall < "$RESULTS_DIR/cpu_flash.txt"
fi

printf "%-14s %12s %14s %12s\n" "Config" "Token/s" "Peak RSS (MB)" "Wall Time"
printf "%-14s %12s %14s %12s\n" "──────────────" "────────────" "──────────────" "────────────"
printf "%-14s %12s %14s %12s\n" "standard" "$std_tps" "$std_rss" "$std_wall"
printf "%-14s %12s %14s %12s\n" "cpu_flash" "$flash_tps" "$flash_rss" "$flash_wall"

# Compute deltas if both are numeric
if [[ "$std_tps" != "N/A" && "$flash_tps" != "N/A" ]]; then
    speed_delta=$(echo "scale=1; ($flash_tps - $std_tps) * 100 / $std_tps" | bc 2>/dev/null) || speed_delta="?"
    echo ""
    echo "Speed: ${speed_delta}% vs standard"
fi
if [[ "$std_rss" != "N/A" && "$flash_rss" != "N/A" ]]; then
    mem_delta=$(echo "scale=1; ($flash_rss - $std_rss) * 100 / $std_rss" | bc 2>/dev/null) || mem_delta="?"
    echo "Memory: ${mem_delta}% vs standard"
fi

echo ""
echo "Raw output in: $RESULTS_DIR/"
