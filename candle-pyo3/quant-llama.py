# This example shows how the candle Python api can be used to replicate llama.cpp.
import sys
from typing import Dict, Tuple, Any
import candle
from candle.models.llama import QuantizedLlama
from candle import utils

MAX_SEQ_LEN = 4096


def gguf_rename(tensor_name: str):
    if tensor_name == "token_embd.weight":
        return "tok_embeddings.weight"
    if tensor_name == "output_norm.weight":
        return "norm.weight"
    tensor_name = tensor_name.replace("blk.", "layers.")
    tensor_name = tensor_name.replace(".attn_q.", ".attention.wq.")
    tensor_name = tensor_name.replace(".attn_k.", ".attention.wk.")
    tensor_name = tensor_name.replace(".attn_v.", ".attention.wv.")
    tensor_name = tensor_name.replace(".attn_output.", ".attention.wo.")
    tensor_name = tensor_name.replace(".ffn_gate.", ".feed_forward.w1.")
    tensor_name = tensor_name.replace(".ffn_down.", ".feed_forward.w2.")
    tensor_name = tensor_name.replace(".ffn_up.", ".feed_forward.w3.")
    tensor_name = tensor_name.replace(".attn_norm.", ".attention_norm.")
    return tensor_name


def main():
    if len(sys.argv) < 2:
        raise ValueError("missing weight file argument")

    filename = sys.argv[1]
    print(f"reading model file {filename}")
    if filename.endswith("gguf"):
        all_tensors, metadata = utils.load_gguf(filename)
        vocab = metadata["tokenizer.ggml.tokens"]
        for i, v in enumerate(vocab):
            vocab[i] = "\n" if v == "<0x0A>" else v.replace("▁", " ")
        hparams = {k: v for (k, v) in metadata.items() if not k.startswith("tokenizer")}
        print(hparams)
        hparams = {
            "n_vocab": len(vocab),
            "n_embd": metadata["llama.embedding_length"],
            "n_mult": 256,
            "n_head": metadata["llama.attention.head_count"],
            "n_head_kv": metadata["llama.attention.head_count_kv"],
            "n_layer": metadata["llama.block_count"],
            "n_rot": metadata["llama.rope.dimension_count"],
            "rope_freq": metadata.get("llama.rope.freq_base", 10000.0),
            "ftype": metadata["general.file_type"],
            "context_length": metadata["llama.context_length"],
        }
        all_tensors = {gguf_rename(k): v for k, v in all_tensors.items()}
    else:
        all_tensors, hparams, vocab = utils.load_ggml(filename)
        hparams["context_length"] = 2048

    print(hparams)
    model = QuantizedLlama(hparams, all_tensors)
    print("model built, starting inference")

    tokens = [1]
    for token_idx in range(500):
        last_token = tokens[-1]
        lt = candle.tensor([last_token]).unsqueeze(0)
        logits = model.forward(lt, len(tokens))
        # Greedy sampling for now
        # pr = candle.nn.softmax(logits, -1)
        m = logits.get(0).argmax_keepdim(-1)
        next_token = m.values()[0]
        print(vocab[next_token], end="", flush=True)
        tokens.append(next_token)


if __name__ == "__main__":
    main()
