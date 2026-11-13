"""Regenerate small Gemma4 golden tensors from pinned upstream PyTorch code.

Development-only fixture generation. Native Candle inference never invokes Python.
Requires torch and numpy; reads upstream sources with their Apache-2.0 headers.
"""
import ast
import hashlib
import json
import math
import types
import urllib.request
import warnings
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REVISION = "109a7d19bf667168837f32929e00b791e195d14d"
BASE = f"https://raw.githubusercontent.com/huggingface/transformers/{REVISION}/src/transformers/"
sources = {}
def source(path):
    data = urllib.request.urlopen(BASE + path).read()
    sources[path] = hashlib.sha256(data).hexdigest()
    return ast.parse(data)

namespace = dict(torch=torch, nn=nn, F=F, math=math, np=np, warnings=warnings,
                 cached_property=cached_property, GradientCheckpointingLayer=nn.Module,
                 ACT2FN={"gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"), "silu": F.silu})
def extract(tree, names):
    nodes = []
    for node in tree.body:
        if getattr(node, "name", None) in names:
            node.decorator_list = []
            nodes.append(node)
    preamble = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[preamble] + nodes, type_ignores=[]))
    exec(compile(module, "pinned-upstream-gemma4", "exec"), namespace)

model_tree = source("models/gemma4/modeling_gemma4.py")
extract(model_tree, {"Gemma4RMSNorm", "Gemma4ClippableLinear", "rotate_half", "apply_rotary_pos_emb",
    "repeat_kv", "eager_attention_forward", "Gemma4TextMLP", "Gemma4TextAttention", "Gemma4TextDecoderLayer",
    "Gemma4AudioRelPositionalEncoding", "Gemma4AudioAttention", "Gemma4AudioSubSampleConvProjectionLayer",
    "Gemma4AudioSubSampleConvProjection", "Gemma4AudioFeedForward", "Gemma4AudioCausalConv1d",
    "Gemma4AudioLightConv1d", "Gemma4AudioLayer"})
namespace["ALL_ATTENTION_FUNCTIONS"] = types.SimpleNamespace(get_interface=lambda _, fallback: fallback)
extract(source("audio_utils.py"), {"hertz_to_mel", "mel_to_hertz", "_create_triangular_filter_bank", "mel_filter_bank", "window_function"})
extract(source("models/gemma4/feature_extraction_gemma4.py"), {"_unfold"})
feature_tree = source("models/gemma4/feature_extraction_gemma4.py")
cls = next(n for n in feature_tree.body if getattr(n, "name", None) == "Gemma4AudioFeatureExtractor")
method = next(n for n in cls.body if getattr(n, "name", None) == "_extract_spectrogram")
module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
exec(compile(module, "pinned-upstream-feature-extractor", "exec"), namespace)

def initialize(module):
    with torch.no_grad():
        for name, tensor in module.state_dict().items():
            phase = sum(name.encode()) * 0.013
            values = torch.sin(torch.arange(tensor.numel()).float() * 0.17 + phase).reshape(tensor.shape)
            if name.endswith("input_min") or name.endswith("output_min"):
                tensor.fill_(-0.3)
            elif name.endswith("input_max") or name.endswith("output_max"):
                tensor.fill_(0.4)
            elif name.endswith("layer_scalar"):
                tensor.fill_(1.07)
            elif "norm" in name and name.endswith("weight"):
                tensor.copy_(1 + values * 0.1)
            else:
                tensor.copy_(values * 0.2)

def weights(module):
    return {name: {"shape": list(t.shape), "values": t.flatten().tolist()} for name, t in module.state_dict().items()}

text_cfg = dict(hidden_size=8, intermediate_size=12, vocab_size=16, num_hidden_layers=4,
    num_attention_heads=2, num_key_value_heads=1, head_dim=4, global_head_dim=4,
    sliding_window=3, layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    rms_norm_eps=1e-6, attention_bias=False, hidden_activation="gelu_pytorch_tanh", attention_dropout=0.,
    hidden_size_per_layer_input=2, vocab_size_per_layer_input=16, num_kv_shared_layers=2,
    use_double_wide_mlp=True, attention_k_eq_v=False, enable_moe_block=False,
    max_position_embeddings=32, tie_word_embeddings=True,
    rope_parameters={"full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 10000.},
                     "sliding_attention": {"rope_theta": 10000.}})
cfg = types.SimpleNamespace(**text_cfg, use_bidirectional_attention=None, _attn_implementation="eager",
    kv_sharing_modes=["store", "store", "reuse", "reuse"],
    per_layer_config={i: types.SimpleNamespace(head_dim=4, num_key_value_heads=1) for i in range(4)})
text = nn.Module()
text.model = nn.Module()
m = text.model
m.embed_tokens = nn.Embedding(16, 8)
m.layers = nn.ModuleList([namespace["Gemma4TextDecoderLayer"](cfg, i) for i in range(4)])
m.norm = namespace["Gemma4RMSNorm"](8)
m.embed_tokens_per_layer = nn.Embedding(16, 8)
m.per_layer_model_projection = nn.Linear(8, 8, bias=False)
m.per_layer_projection_norm = namespace["Gemma4RMSNorm"](2)
initialize(text)
text.eval()
class Cache:
    def __init__(self): self.kv = {}
    def update(self, k, v, layer):
        if layer in self.kv:
            old_k, old_v = self.kv[layer]
            k, v = torch.cat([old_k, k], 2), torch.cat([old_v, v], 2)
        self.kv[layer] = k, v
        return k, v
cache = Cache()
def text_forward(ids, offset):
    ids = torch.tensor([ids])
    xs = m.embed_tokens(ids) * math.sqrt(8)
    identity = (m.embed_tokens_per_layer(ids) * math.sqrt(2)).reshape(1, -1, 4, 2)
    context = m.per_layer_projection_norm((m.per_layer_model_projection(xs) / math.sqrt(8)).reshape(1, -1, 4, 2))
    per_layer = (identity + context) / math.sqrt(2)
    q_len = ids.shape[1]
    positions = torch.arange(offset, offset + q_len)
    shared = {}
    for i, layer in enumerate(m.layers):
        sliding = cfg.layer_types[i] == "sliding_attention"
        inv = torch.tensor([1., 1 / 100.]) if sliding else torch.tensor([1., 0.])
        angles = positions[:, None] * inv
        angles = torch.cat([angles, angles], -1).unsqueeze(0)
        keys = torch.arange(offset + q_len)
        valid = keys[None, :] <= positions[:, None]
        if sliding: valid &= keys[None, :] + 3 > positions[:, None]
        mask = torch.where(valid, 0., float("-inf"))[None, None, :, :]
        xs = layer(xs, per_layer[:, :, i], shared_kv_states=shared, past_key_values=cache,
                   position_embeddings=(angles.cos().to(xs.dtype), angles.sin().to(xs.dtype)), attention_mask=mask.to(xs.dtype))
    return F.linear(m.norm(xs[:, -1]), m.embed_tokens.weight).detach().flatten().tolist()
with torch.no_grad():
    text_cases = [{"ids": ids, "offset": offset, "expected": text_forward(ids, offset)} for ids, offset in [([1,3,4,7,9,6],0),([2],6),([5,8],7)]]
text_weights = weights(text)
text.to(torch.bfloat16)
cache = Cache()
with torch.no_grad():
    for case in text_cases:
        case["bf16_expected"] = text_forward(case["ids"], case["offset"])

audio_cfg = dict(input_feat_size=4, hidden_size=8, output_proj_dims=4,
    attention_chunk_size=4, attention_context_left=5, attention_context_right=0,
    attention_invalid_logits_value=-1e9, attention_logit_cap=50., num_attention_heads=2,
    num_hidden_layers=1, conv_kernel_size=3, residual_weight=0.5, rms_norm_eps=1e-6,
    subsampling_conv_channels=[4,2], hidden_act="silu", gradient_clipping=1e10, use_clipped_linears=True)
acfg = types.SimpleNamespace(**audio_cfg)
audio = nn.Module()
audio.subsample_conv_projection = namespace["Gemma4AudioSubSampleConvProjection"](acfg)
audio.layers = nn.ModuleList([namespace["Gemma4AudioLayer"](acfg,0)])
audio.output_proj = nn.Linear(8,4)
initialize(audio)
audio.eval()
mel = (torch.arange(9 * 4).float() * 0.19).sin().reshape(1,9,4)
valid = torch.tensor([[True]*7 + [False]*2])
with torch.no_grad():
    features, output_valid = audio.subsample_conv_projection(mel,valid)
    t = features.shape[1]
    blocks = (t+3)//4
    mask = torch.full((1,1,blocks,4,8), -1e9)
    for block in range(blocks):
        for q in range(4):
            for k in range(8):
                qpos = block*4+q
                kpos = block*4+k-4
                if 0 <= qpos < t and 0 <= kpos < t and output_valid[0,kpos] and 0 <= qpos-kpos < 4:
                    mask[0,0,block,q,k] = 0
    pos = namespace["Gemma4AudioRelPositionalEncoding"](acfg)(features)
    for layer in audio.layers: features = layer(features, mask, pos)
    features = audio.output_proj(features) * output_valid[...,None]
audio_weights = weights(audio)
audio.to(torch.bfloat16)
with torch.no_grad():
    bf16_features, bf16_valid = audio.subsample_conv_projection(mel.to(torch.bfloat16),valid)
    pos = namespace["Gemma4AudioRelPositionalEncoding"](acfg)(bf16_features)
    for layer in audio.layers: bf16_features = layer(bf16_features, mask, pos)
    bf16_features = audio.output_proj(bf16_features) * bf16_valid[...,None]

extractor = types.SimpleNamespace(dither=0., input_scale_factor=1., frame_length=320, hop_length=160,
    preemphasis=0., preemphasis_htk_flavor=True, fft_length=512, mel_floor=np.array(0.001),
    per_bin_mean=None, per_bin_stddev=None, window=namespace["window_function"](320).astype(np.float32),
    mel_filters=namespace["mel_filter_bank"](257,128,0.,8000.,16000,norm=None,mel_scale="htk"))
feature_cases=[]
for count in [161,640,1050]:
    samples=(np.sin(np.arange(count)*0.071)*0.3).astype(np.float32)
    padded=(count+127)//128*128
    wave=np.pad(samples,(0,padded-count))[None,:]
    mask=np.arange(padded)<count
    mel, valid=namespace["_extract_spectrogram"](extractor,wave,mask)
    mel=(mel*valid[:,None]).astype(np.float32)
    feature_cases.append(dict(samples=samples.tolist(),shape=list(mel.shape),expected=mel.flatten().tolist(),padding_mask=(~valid).astype(float).tolist()))

out = dict(revision=REVISION, source_sha256=sources, torch=torch.__version__, text_config=text_cfg,
    text_weights=text_weights, text_cases=text_cases, audio_config=audio_cfg, audio_weights=audio_weights,
    audio_input=(torch.arange(36).float()*0.19).sin().tolist(),
    audio_padding_mask=[0.]*7+[1.]*2, audio_expected=features.flatten().tolist(), audio_shape=list(features.shape),
    audio_output_padding_mask=(~output_valid).float().flatten().tolist(), feature_cases=feature_cases)
out["audio_bf16_expected"] = bf16_features.flatten().tolist()
# Voxtral uses WhisperFeatureExtractor, not whisper.cpp's legacy framing.
whisper_tree = source("models/whisper/feature_extraction_whisper.py")
whisper_cls = next(n for n in whisper_tree.body if getattr(n,"name",None)=="WhisperFeatureExtractor")
whisper_method = next(n for n in whisper_cls.body if getattr(n,"name",None)=="_torch_extract_fbank_features")
exec(compile(ast.fix_missing_locations(ast.Module(body=[whisper_method],type_ignores=[])),"pinned-upstream-whisper-features","exec"),namespace)
filters = np.frombuffer((Path(__file__).parents[2]/"assets/melfilters128.bytes").read_bytes(),dtype="<f4").reshape(128,201).T.copy()
vox_samples = (np.sin(np.arange(1601)*0.053)*0.4).astype(np.float32)
vox_padded = np.pad(vox_samples,(0,480000-len(vox_samples)))
vox_features = namespace["_torch_extract_fbank_features"](types.SimpleNamespace(n_fft=400,hop_length=160,dither=0.,mel_filters=filters),vox_padded)
out["voxtral_features"] = dict(samples=vox_samples.tolist(),shape=list(vox_features.shape),
    points=[dict(band=band,frame=frame,value=float(vox_features[band,frame])) for band in [0,1,12,63,127] for frame in [0,1,3,10,2998,2999]])
out["source_sha256"] = sources
Path(__file__).with_suffix(".json").write_text(json.dumps(out,indent=2)+"\n")
