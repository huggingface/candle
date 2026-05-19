# wav2vec2

[Wav2Vec2](https://arxiv.org/abs/2006.11477) is a self-supervised speech
representation learning model by Facebook AI.  This example runs
`Wav2Vec2ForCTC` — the CTC-fine-tuned variant — for speech-to-text.

## Quick start

```bash
# Transcribe a 16 kHz mono WAV file using facebook/wav2vec2-base-960h
cargo run --example wav2vec2 --release -- \
    --model facebook/wav2vec2-base-960h \
    --audio  path/to/audio.wav
```

## Weight-norm preprocessing

HuggingFace wav2vec2 checkpoints store the positional convolutional embedding
with `nn.utils.weight_norm` parametrization.  Before loading into Candle you
must export a clean safetensors file with the parametrization removed:

```python
from transformers import Wav2Vec2ForCTC
from safetensors.torch import save_file
import torch

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()
torch.nn.utils.parametrize.remove_parametrizations(
    model.wav2vec2.encoder.pos_conv_embed.conv, "weight")
save_file(
    {k: v.float() for k, v in model.state_dict().items()
     if "parametrizations" not in k},
    "wav2vec2_clean.safetensors")
```

Then pass the clean file with `--weights`:

```bash
cargo run --example wav2vec2 --release -- \
    --model-dir /path/to/wav2vec2-base-960h \
    --weights   wav2vec2_clean.safetensors \
    --audio     path/to/audio.wav
```

## Output

```
Device: Cpu
Config: hidden=768 layers=12 heads=12 ffn=3072 cnn_norm=group do_stable_ln=false
Model loaded.
Vocab size: 32
Audio: 66048 samples (4.13s)
Inference: 1240ms  (logits [1, 205, 32])

Transcription:
THE QUICK BROWN FOX
```

## Architecture

```
Wav2Vec2ForCTC
├── Feature extractor  7× Conv1d (no-pad, stride=[5,2,2,2,2,2,2])
│   layer 0: InstanceNorm + GELU  (feat_extract_norm="group")
│   layers 1-6: GELU only
├── Feature projection LayerNorm → Linear
├── Positional conv    grouped Conv1d (K=128, G=16) + GELU + residual
├── L× Encoder layer   pre-norm MHA + GELU FFN
├── Global LayerNorm   applied *after* all transformer layers
└── LM head            Linear → CTC greedy decode
```

The global LayerNorm placement (after all transformer layers, not before) is
the key correctness detail in `Wav2Vec2EncoderStableLayerNorm`.
