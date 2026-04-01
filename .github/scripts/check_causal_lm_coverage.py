#!/usr/bin/env python3
"""
CI check: every model file that looks like a causal LM (has a public `forward`
accepting a `seqlen_offset` / `index_pos` position argument) must either:

  1. Call `impl_causal_lm!` or `causal_lm_wrapper!` (registered in AutoModelForCausalLM), OR
  2. Appear in the ALLOWLIST below with a documented reason.

Run locally:
    python3 .github/scripts/check_causal_lm_coverage.py
"""

import re
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "candle-transformers/src/models"

# ── ALLOWLIST ──────────────────────────────────────────────────────────────────
# Files that contain a position-offset `forward` but are NOT autoregressive
# text generators, or require a custom ForCausalLM wrapper not yet implemented.
ALLOWLIST = {
    # ── Non-LLM models ────────────────────────────────────────────────────────
    "based.rs":         "Based (attention+SSM hybrid), not a pure causal LM",
    "csm.rs":           "Conversational Speech Model — audio, not text",
    "embedding.rs":     "Text embedding model — encoder, not decoder",
    "parler_tts.rs":    "Parler TTS — text-to-speech, not a text LM",

    # ── Sub-module / shared utility files ────────────────────────────────────
    "mod.rs":           "Module file, not a model implementation",
    "model.rs":         "Shared model utilities",
    "text.rs":          "Sub-module used by multi-modal models",
    "transformer.rs":   "Generic transformer sub-module",

    # ── External-cache pattern: needs ForCausalLM wrapper (TODO) ─────────────
    "granitemoehybrid.rs":
        "TODO: uses external GraniteMoeHybridCache, needs causal_lm_wrapper!",
    "llama2_c.rs":
        "TODO: Karpathy llama2.c variant with external Cache, needs causal_lm_wrapper!",
    "quantized_llama2_c.rs":
        "TODO: quantized llama2.c with external Cache, needs causal_lm_wrapper!",

    # ── Voxtral: speech+language model, speech-specific forward ──────────────
    "voxtral_llama.rs":
        "Voxtral uses LLaMA backbone but is wired through the Voxtral model wrapper",
}

# ── Heuristic: does this file look like a causal LM? ─────────────────────────
# A file looks like a causal LM if its top-level pub forward takes a position arg.
POSITION_ARG_RE = re.compile(
    r"pub fn forward\s*\([^)]*(?:seqlen_offset|index_pos|_seqlen_offset)\s*:\s*usize"
)

REGISTERED_RE = re.compile(r"impl_causal_lm!|causal_lm_wrapper!")


def check() -> int:
    errors: list[str] = []
    warned: list[str] = []

    for path in sorted(MODELS_DIR.rglob("*.rs")):
        name = path.name

        # Skip files explicitly on the allowlist
        if name in ALLOWLIST:
            continue

        src = path.read_text(errors="replace")

        looks_like_lm = bool(POSITION_ARG_RE.search(src))
        is_registered = bool(REGISTERED_RE.search(src))

        if looks_like_lm and not is_registered:
            rel = str(path.relative_to(MODELS_DIR))
            errors.append(
                f"  {rel}\n"
                f"    → has seqlen_offset/index_pos in forward but no impl_causal_lm! or\n"
                f"      causal_lm_wrapper!. Add the macro OR add to the ALLOWLIST with a\n"
                f"      reason."
            )

    if errors:
        print(
            "❌  AutoModelForCausalLM coverage check FAILED\n"
            "    The following model files look like causal LMs but are not registered:\n"
        )
        for e in errors:
            print(e)
        print(
            "\n    Fix by adding one of:\n"
            "      crate::impl_causal_lm!(MyModel, \"model_type\");\n"
            "      crate::causal_lm_wrapper!(MyForCausalLM, \"model_type\", ...);\n"
            "\n    Or add the file to ALLOWLIST in .github/scripts/check_causal_lm_coverage.py\n"
            "    with a documented reason why it should be skipped."
        )
        return 1

    print(f"✅  AutoModelForCausalLM coverage check passed ({len(list(MODELS_DIR.rglob('*.rs')))} files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(check())
