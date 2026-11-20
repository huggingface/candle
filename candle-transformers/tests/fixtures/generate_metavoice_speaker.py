"""Regenerate independent speaker fixtures with public weights, PyTorch and Librosa.

Example: python generate_metavoice_speaker.py --checkpoint speaker_encoder.safetensors
  --wav candle-audio/tests/fixtures/question-en.wav --output metavoice_speaker_encoder.json
This utility is test-only; native Candle inference does not invoke Python.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import types
import urllib.request
import wave

import librosa
import numpy as np
import torch
from safetensors.torch import load_file

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--wav", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
revision = "de3fa211ac4621e03a5f990651aeecc64da418f5"
base = f"https://raw.githubusercontent.com/metavoiceio/metavoice-src/{revision}/fam/quantiser/audio/speaker_encoder/"
model_source = urllib.request.urlopen(base + "model.py").read()
audio_source = urllib.request.urlopen(base + "audio.py").read()
audio_namespace = {}
exec(compile(audio_source, base + "audio.py", "exec"), audio_namespace)
namespace = dict(torch=torch, np=np, audio=types.SimpleNamespace(**audio_namespace),
                 sampling_rate=16000, mel_window_step=10, partials_n_frames=160)
model_ast = ast.parse(model_source)
reference_class = next(node for node in model_ast.body if isinstance(node, ast.ClassDef))
for node in reference_class.body:
    if isinstance(node, ast.FunctionDef) and node.name in ["forward", "compute_partial_slices", "embed_utterance"]:
        exec(compile(ast.Module(body=[node], type_ignores=[]), base + "model.py", "exec"), namespace)

class Reference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(40, 256, 3, batch_first=True)
        self.linear = torch.nn.Linear(256, 256)
        self.relu = torch.nn.ReLU()
        self.device = torch.device("cpu")

for name in ["forward", "compute_partial_slices", "embed_utterance"]:
    setattr(Reference, name, namespace[name])
torch.set_num_threads(1)
model = Reference().eval()
model.load_state_dict(load_file(args.checkpoint), strict=False)
with wave.open(args.wav, "rb") as wav:
    assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
    samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
with torch.no_grad():
    embedding = model.embed_utterance(samples)
pcm = np.array([0.125 * np.cos(2 * np.pi * 175 * i / 16000) + 0.03 * np.sin(2 * np.pi * 3300 * i / 16000)
                for i in range(640)], dtype=np.float32)
mel = audio_namespace["wav_to_mel_spectrogram"](pcm)
slices = []
for length in [1000, 25600, 30000, 100000]:
    wav_slices, mel_slices = model.compute_partial_slices(length, 1.3, 0.75)
    slices.append(dict(samples=length, wav=[[int(s.start), int(s.stop)] for s in wav_slices],
                       mel=[[int(s.start), int(s.stop)] for s in mel_slices]))
result = dict(source=base + "model.py", source_sha256=hashlib.sha256(model_source).hexdigest(),
              frontend_source=base + "audio.py", frontend_sha256=hashlib.sha256(audio_source).hexdigest(),
              torch=torch.__version__, librosa=librosa.__version__, numpy=np.__version__,
              speaker_checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              waveform_sha256=hashlib.sha256(Path(args.wav).read_bytes()).hexdigest(),
              pcm=pcm.tolist(), mel_shape=list(mel.shape), mel=mel.reshape(-1).tolist(),
              slices=slices, embedding=embedding.tolist())
Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
