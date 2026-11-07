The experimental Qwen3.5 backbone in `src/qwen3_5.rs` derives from woodRock's
Candle pull request #3396, commit e9e4929f3e31b6183cc17e7be0596929c6725257.
Source: https://github.com/woodRock/candle/blob/e9e4929f3e31b6183cc17e7be0596929c6725257/candle-transformers/src/models/qwen3_5.rs

That code is licensed under MIT or Apache-2.0, as is Candle. Copies of both
licenses are included in this directory. The local modifications add partial
rotary support, multi-token cache continuation, and fused CPU kernels.

Jared Palmer's Kev source and released model are Apache-2.0. The clone remains
in `../kev` with its original license. Model weights are downloaded separately
and excluded from version control. The implementation follows Kev's input
format and pointer scoring semantics; Python scripts import the official Kev
code for reference validation.
