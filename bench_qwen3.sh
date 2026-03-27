#!/usr/bin/env bash
# Benchmark Qwen3-0.6B: Standard CPU vs CPU Flash Attention
#
# Measures token/s throughput and peak RSS (memory) across prompt sizes.
# Runs each config N times and reports mean ± stddev.
#
# Usage:
#   ./bench_qwen3.sh [sample_len] [runs] [mode]
#   mode: all (default), decode, prefill
#
# Requirements:
#   - macOS (uses /usr/bin/time -l for peak RSS)
#   - Model weights auto-downloaded on first run (~1.2 GB for Qwen3-0.6B)

set -euo pipefail

SAMPLE_LEN="${1:-400}"
RUNS="${2:-3}"
MODE="${3:-all}"
MODEL="3-0.6b"

# ── Prompts at different lengths ─────────────────────────────────────────

SHORT_PROMPT="Explain the theory of relativity in simple terms."

# ~500 tokens: opening of Ulysses
MEDIUM_PROMPT="Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of \
lather on which a mirror and a razor lay crossed. A yellow dressinggown, ungirdled, was \
sustained gently behind him on the mild morning air. He held the bowl aloft and intoned: \
Introibo ad altare Dei. Halted, he peered down the dark winding stairs and called out \
coarsely: Come up, Kinch! Come up, you fearful jesuit! Solemnly he came forward and \
mounted the round gunrest. He faced about and blessed gravely thrice the tower, the \
surrounding land and the awaking mountains. Then, catching sight of Stephen Dedalus, he \
bent towards him and made rapid crosses in the air, gurgling in his throat and shaking \
his head. Stephen Dedalus, displeased and sleepy, leaned his arms on the top of the \
staircase and looked coldly at the shaking gurgling face that blessed him, equine in its \
length, and at the light untonsured hair, grained and hued like pale oak. Buck Mulligan \
peeped an instant under the mirror and then covered the bowl smartly. Back to barracks! \
he said sternly. He added in a preacher's tone: For this, O dearly beloved, is the \
genuine Christine: body and soul and blood and ouns. Slow music, please. Shut your eyes, \
gents. One moment. A little trouble about those white corpuscles. Silence, all. He peered \
sideways up and gave a long slow whistle of call, then paused awhile in rapt attention, \
his even white teeth glistening here and there with gold points. Chrysostomos. Two strong \
shrill whistles answered through the calm. He turned to Stephen and said gravely: The \
mockery of it! Your absurd name, an ancient Greek! He pointed his finger in friendly jest \
and went over to the parapet, laughing to himself. Stephen Dedalus stepped up, followed \
him wearily halfway and sat down on the edge of the gunrest, watching him still as he \
propped his mirror on the parapet, dipped the brush in the bowl and lathered cheeks and \
neck. Buck Mulligan's gay voice went on. My name is absurd too: Malachi Mulligan, two \
dactyls. But it has a Hellenic ring, hasn't it? Tripping and sunny like the buck himself. \
We must go to Athens. Will you come if I can get the aunt to fork out twenty quid? He laid \
the brush aside and, laughing with delight, cried: Will he come? The jejune jesuit! \
Ceasing, he began to shave with care. Tell me, Mulligan, Stephen said quietly. Yes, my \
love? How long is Haines going to stay in this tower? Buck Mulligan showed a shaven cheek \
over his right shoulder. God, isn't he dreadful? he said frankly. A ponderous Saxon. He \
thinks you're not a gentleman. God, these bloody English! Bursting with money and \
indigestion. Because he comes from Oxford. You know, Dedalus, you have the real Oxford \
manner. He can't make you out. O, my name for you is the best: Kinch, the knifeblade. \
He shaved warily over his chin. He was raving all night about a black panther, Stephen \
said. Edward. Is it Haines his name? That beastly fellow? Mulligan turned from the sea \
to Stephen. He looked him full in the face and went on."

# ~2000 tokens: extended Ulysses (chapters 1-2 opening)
LONG_PROMPT="Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of \
lather on which a mirror and a razor lay crossed. A yellow dressinggown, ungirdled, was \
sustained gently behind him on the mild morning air. He held the bowl aloft and intoned: \
Introibo ad altare Dei. Halted, he peered down the dark winding stairs and called out \
coarsely: Come up, Kinch! Come up, you fearful jesuit! Solemnly he came forward and \
mounted the round gunrest. He faced about and blessed gravely thrice the tower, the \
surrounding land and the awaking mountains. Then, catching sight of Stephen Dedalus, he \
bent towards him and made rapid crosses in the air, gurgling in his throat and shaking \
his head. Stephen Dedalus, displeased and sleepy, leaned his arms on the top of the \
staircase and looked coldly at the shaking gurgling face that blessed him, equine in its \
length, and at the light untonsured hair, grained and hued like pale oak. Buck Mulligan \
peeped an instant under the mirror and then covered the bowl smartly. Back to barracks! \
he said sternly. He added in a preacher's tone: For this, O dearly beloved, is the \
genuine Christine: body and soul and blood and ouns. Slow music, please. Shut your eyes, \
gents. One moment. A little trouble about those white corpuscles. Silence, all. He peered \
sideways up and gave a long slow whistle of call, then paused awhile in rapt attention, \
his even white teeth glistening here and there with gold points. Chrysostomos. Two strong \
shrill whistles answered through the calm. He turned to Stephen and said gravely: The \
mockery of it! Your absurd name, an ancient Greek! He pointed his finger in friendly jest \
and went over to the parapet, laughing to himself. Stephen Dedalus stepped up, followed \
him wearily halfway and sat down on the edge of the gunrest, watching him still as he \
propped his mirror on the parapet, dipped the brush in the bowl and lathered cheeks and \
neck. Buck Mulligan's gay voice went on. My name is absurd too: Malachi Mulligan, two \
dactyls. But it has a Hellenic ring, hasn't it? Tripping and sunny like the buck himself. \
We must go to Athens. Will you come if I can get the aunt to fork out twenty quid? He laid \
the brush aside and, laughing with delight, cried: Will he come? The jejune jesuit! \
Ceasing, he began to shave with care. Tell me, Mulligan, Stephen said quietly. Yes, my \
love? How long is Haines going to stay in this tower? Buck Mulligan showed a shaven cheek \
over his right shoulder. God, isn't he dreadful? he said frankly. A ponderous Saxon. He \
thinks you're not a gentleman. God, these bloody English! Bursting with money and \
indigestion. Because he comes from Oxford. You know, Dedalus, you have the real Oxford \
manner. He can't make you out. O, my name for you is the best: Kinch, the knifeblade. \
He shaved warily over his chin. He was raving all night about a black panther, Stephen \
said. Edward. Is it Haines his name? That beastly fellow? Mulligan turned from the sea \
to Stephen. He looked him full in the face and went on. \
The aunt thinks you killed your mother, he said. That's why she won't let me have \
anything to do with you. She also knew, or had been told, something else. I told her she \
was wrong. She doesn't know anything about it, does she? Mulligan said. She thinks I \
did it, Stephen said. And yet you refused to pray for your dying mother when she begged \
you to. Isn't that so? Buck Mulligan swung round on his heel. You could have knelt down, \
damn it, Kinch, when your dying mother asked you, Buck Mulligan said. I'm hyperborean \
as much as you. But to think of your mother begging you with her last breath to kneel \
down and pray for her. And you refused. There is something sinister in you. He broke \
off and lathered again lightly his farther cheek. A tolerant smile curled his lips. But \
a lovely mummer, he murmured to himself. Kinch, the loveliest mummer of them all. He \
shaved evenly and with care, in silence, seriously. Stephen, an elbow rested on the \
jagged granite, leaned his palm against his brow and gazed at the fraying edge of his \
shiny black coatsleeve. Pain, that was not yet the pain of love, fretted his heart. \
Silently, in a dream she had come to him after her death, her wasted body within its \
loose brown graveclothes giving off an odour of wax and rosewood, her breath, that had \
bent upon him, mute, reproachful, a faint odour of wetted ashes. Across the threadbare \
cuffedge he saw the sea hailed as a great sweet mother by the wellfed voice beside him. \
The ring of bay and skyline held a dull green mass of liquid. A bowl of white china had \
stood beside her deathbed holding the green sluggish bile which she had torn up from her \
rotting liver by fits of loud groaning vomiting. Buck Mulligan wiped the razorblade \
neatly. Then he gazed over the handkerchief, folding it as he gazed, and said: The \
bard's noserag. A new art colour for our Irish poets: snotgreen. You can almost taste \
it, can't you? He mounted to the parapet again and gazed out over Dublin bay, his fair \
oakpale hair stirring slightly. God, he said quietly. Isn't the sea what Algy calls it: \
a grey sweet mother? The snotgreen sea. The scrotumtightening sea. Epi oinopa ponton. \
Ah, Dedalus, the Greeks. I must teach you. You must read them in the original. Thalatta! \
Thalatta! She is our great sweet mother. Come and look. Stephen stood up and went over \
to the parapet. Leaning on it he looked down on the water and on the mailboat clearing \
the harbourmouth of Kingstown. Our mighty mother, Buck Mulligan said. He turned abruptly \
his great searching eyes from the sea to Stephen's face. The aunt thinks you killed your \
mother, he said. That's why she won't let me have anything to do with you. Someone \
killed her, Stephen said gloomily. You could have knelt down, damn it, Kinch, when your \
dying mother asked you, Buck Mulligan said. I'm hyperborean as much as you. But to think \
of your mother begging you with her last breath to kneel down and pray for her. And you \
refused. There is something sinister in you. He broke off and lathered his farther cheek. \
A tolerant smile curled his lips again. But a lovely mummer, he murmured to himself."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/bench_results"
BINARY="$SCRIPT_DIR/target/release/examples/qwen"

mkdir -p "$RESULTS_DIR"

# ── Run one invocation, write raw metrics to a file ──────────────────────
run_once() {
    local name="$1"
    local use_flash="$2"
    local run_idx="$3"
    local raw_file="$RESULTS_DIR/${name}_run${run_idx}.txt"

    local flash_flag=""
    if [[ "$use_flash" == "true" ]]; then
        flash_flag="--use-flash-attn"
    fi

    /usr/bin/time -l "$BINARY" \
        --cpu \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --sample-len "$SAMPLE_LEN" \
        $flash_flag \
        2>"$RESULTS_DIR/${name}_run${run_idx}_time.txt" \
        >"$RESULTS_DIR/${name}_run${run_idx}_stdout.txt" \
        || true

    local tps rss
    tps=$(grep -oE '[0-9]+\.[0-9]+ token/s' "$RESULTS_DIR/${name}_run${run_idx}_stdout.txt" \
        | tail -1 | grep -oE '[0-9]+\.[0-9]+') || tps="0"
    rss=$(grep "maximum resident set size" "$RESULTS_DIR/${name}_run${run_idx}_time.txt" \
        | grep -oE '[0-9]+') || rss="0"

    # Convert RSS bytes to MB
    if [[ "$rss" -gt 0 ]]; then
        rss=$(echo "scale=1; $rss / 1048576" | bc)
    fi

    echo "$tps $rss" > "$raw_file"
}

# ── Run N times, compute mean ± stddev ───────────────────────────────────
run_bench() {
    local name="$1"
    local use_flash="$2"

    echo "    $name (${RUNS}x) ..."

    local tps_vals=""
    local rss_vals=""

    for i in $(seq 1 "$RUNS"); do
        run_once "$name" "$use_flash" "$i"
        read -r tps rss < "$RESULTS_DIR/${name}_run${i}.txt"
        tps_vals="$tps_vals $tps"
        rss_vals="$rss_vals $rss"
    done

    # Compute mean and stddev with awk
    local tps_stats rss_stats
    tps_stats=$(echo "$tps_vals" | awk '{
        n=NF; sum=0; sumsq=0
        for(i=1;i<=n;i++){sum+=$i; sumsq+=$i*$i}
        mean=sum/n
        var=(n>1) ? (sumsq/n - mean*mean) : 0
        if(var<0) var=0
        printf "%.2f %.2f", mean, sqrt(var)
    }')
    rss_stats=$(echo "$rss_vals" | awk '{
        n=NF; sum=0; sumsq=0
        for(i=1;i<=n;i++){sum+=$i; sumsq+=$i*$i}
        mean=sum/n
        var=(n>1) ? (sumsq/n - mean*mean) : 0
        if(var<0) var=0
        printf "%.1f %.1f", mean, sqrt(var)
    }')

    local tps_mean tps_sd rss_mean rss_sd
    tps_mean=$(echo "$tps_stats" | awk '{print $1}')
    tps_sd=$(echo "$tps_stats" | awk '{print $2}')
    rss_mean=$(echo "$rss_stats" | awk '{print $1}')
    rss_sd=$(echo "$rss_stats" | awk '{print $2}')

    echo "$tps_mean $tps_sd $rss_mean $rss_sd" > "$RESULTS_DIR/${name}.txt"
}

# ── Print comparison table ───────────────────────────────────────────────
print_results() {
    local label="$1"
    local std_name="$2"
    local flash_name="$3"

    local std_tps_mean="N/A" std_tps_sd="0" std_rss_mean="N/A" std_rss_sd="0"
    local flash_tps_mean="N/A" flash_tps_sd="0" flash_rss_mean="N/A" flash_rss_sd="0"

    if [[ -f "$RESULTS_DIR/${std_name}.txt" ]]; then
        read -r std_tps_mean std_tps_sd std_rss_mean std_rss_sd < "$RESULTS_DIR/${std_name}.txt"
    fi
    if [[ -f "$RESULTS_DIR/${flash_name}.txt" ]]; then
        read -r flash_tps_mean flash_tps_sd flash_rss_mean flash_rss_sd < "$RESULTS_DIR/${flash_name}.txt"
    fi

    echo "$label"
    echo ""
    printf "  %-14s %18s %20s\n" "Config" "Token/s (mean±sd)" "Peak RSS MB (mean±sd)"
    printf "  %-14s %18s %20s\n" "──────────────" "──────────────────" "─────────────────────"
    printf "  %-14s %10s ± %-5s %12s ± %-5s\n" "standard" "$std_tps_mean" "$std_tps_sd" "$std_rss_mean" "$std_rss_sd"
    printf "  %-14s %10s ± %-5s %12s ± %-5s\n" "cpu_flash" "$flash_tps_mean" "$flash_tps_sd" "$flash_rss_mean" "$flash_rss_sd"

    if [[ "$std_tps_mean" != "N/A" && "$flash_tps_mean" != "N/A" && "$std_tps_mean" != "0" ]]; then
        local speed_delta
        speed_delta=$(echo "scale=1; ($flash_tps_mean - $std_tps_mean) * 100 / $std_tps_mean" | bc 2>/dev/null) || speed_delta="?"
        echo ""
        echo "  Speed:  ${speed_delta}% vs standard"
    fi
    if [[ "$std_rss_mean" != "N/A" && "$flash_rss_mean" != "N/A" && "$std_rss_mean" != "0" ]]; then
        local mem_delta
        mem_delta=$(echo "scale=1; ($flash_rss_mean - $std_rss_mean) * 100 / $std_rss_mean" | bc 2>/dev/null) || mem_delta="?"
        echo "  Memory: ${mem_delta}% vs standard"
    fi
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────────
echo "=== Qwen3-0.6B CPU Attention Benchmark ==="
echo ""
echo "Config: sample_len=$SAMPLE_LEN, runs=$RUNS (mean±sd), mode=$MODE, model=$MODEL"
echo "Commit: $(git rev-parse --short HEAD)"
echo ""

echo "Step 1: Building release binary"
cargo build --release --example qwen -p candle-examples 2>&1 | tail -3
echo ""

echo "Step 2: Running benchmarks ($RUNS runs each, reporting mean ± sd)"
echo ""

if [[ "$MODE" == "all" || "$MODE" == "decode" ]]; then
    echo "  [Decode] short prompt, generate $SAMPLE_LEN tokens"
    PROMPT="$SHORT_PROMPT"
    run_bench "decode_standard"  "false"
    run_bench "decode_cpu_flash" "true"
    echo ""
fi

if [[ "$MODE" == "all" || "$MODE" == "prefill" ]]; then
    echo "  [Prefill ~500 tok] long prompt, generate $SAMPLE_LEN tokens"
    PROMPT="$MEDIUM_PROMPT"
    run_bench "prefill500_standard"  "false"
    run_bench "prefill500_cpu_flash" "true"
    echo ""

    echo "  [Prefill ~2000 tok] very long prompt, generate $SAMPLE_LEN tokens"
    PROMPT="$LONG_PROMPT"
    run_bench "prefill2k_standard"  "false"
    run_bench "prefill2k_cpu_flash" "true"
    echo ""
fi

echo "=== Results (mean ± stddev, n=$RUNS) ==="
echo ""
if [[ "$MODE" == "all" || "$MODE" == "decode" ]]; then
    print_results "Decode (short prompt):" "decode_standard" "decode_cpu_flash"
fi
if [[ "$MODE" == "all" || "$MODE" == "prefill" ]]; then
    print_results "Prefill ~500 tokens:" "prefill500_standard" "prefill500_cpu_flash"
    print_results "Prefill ~2000 tokens:" "prefill2k_standard" "prefill2k_cpu_flash"
fi

echo "Raw output in: $RESULTS_DIR/"
