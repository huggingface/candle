# candle-muscriptor: audio-to-MIDI music transcription

MuScriptor is a decoder-only causal transformer (audiocraft/MusicGen lineage)
that transcribes audio to MIDI. Audio is processed in 5-second chunks: each
chunk's log-mel spectrogram is projected and prepended as prefix tokens
(together with instrument-group / dataset class embeddings), after which MIDI
event tokens are generated autoregressively and decoded into notes.

Three variants are published on the HuggingFace hub — note the weights are
licensed CC BY-NC 4.0 (non-commercial):

- `small` (768d / 14 layers)
- `medium` (1024d / 24 layers, the default)
- `large` (1536d / 48 layers)

## Running an example

```bash
cargo run --example muscriptor --release --features symphonia -- audio.mp3
```

On Apple Silicon, add `--features metal,accelerate`; the transformer then runs
in f16 on the GPU by default (use `--dtype f32` to force full precision).

Useful flags:

```bash
# model size or a local safetensors path (config.json read from alongside)
--model large
# write somewhere specific instead of <audio>.mid
-o out.mid
# print note events as they stream
--notes
# condition the model on the instruments present in the recording
--instruments acoustic_piano,drums
# temperature sampling instead of greedy decoding
--sampling -t 0.9 --top-p 0.95
# chunks transcribed per batch (default: 4 on GPU, 1 on CPU)
--batch-size 8
```

The tokenizer recovers onsets, offsets, pitch and instrument group — not
velocity (all notes are written at velocity 100).
