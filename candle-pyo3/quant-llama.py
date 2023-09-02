# This example shows how the candle Python api can be used to replicate llama.cpp.
import os
import sys

# The "import candle" statement below works if there is a "candle.so" file in sys.path.
# Here we check for shared libraries that can be used in the build directory.
BUILD_DIR = "./target/release-with-debug"
so_file = BUILD_DIR + "/candle.so"
if os.path.islink(so_file): os.remove(so_file)
for lib_file in ["libcandle.dylib", "libcandle.so"]:
    lib_file_ = BUILD_DIR + "/" + lib_file
    if os.path.isfile(lib_file_):
        os.symlink(lib_file, so_file)
        sys.path.insert(0, BUILD_DIR)
        break

import candle

class RmsNorm:
    def __init__(self, qtensor):
        self.weight = qtensor.dequantize()

    def __call__(self, x):
        b_size, seq_len, hidden_size = x.shape
        norm_x = x.sqr().sum_keepdim(2) / hidden_size
        x_normed = x / (norm_x + 1e-5).sqrt()
        return x_normed * self.weight

class QuantizedLayer:
    def __init__(self, layer_idx, all_tensors):
        p = f"layers.{layer_idx}"
        self.attention_wq = all_tensors[f"{p}.attention.wq.weight"]
        self.attention_wk = all_tensors[f"{p}.attention.wk.weight"]
        self.attention_wv = all_tensors[f"{p}.attention.wv.weight"]
        self.attention_wo = all_tensors[f"{p}.attention.wo.weight"]
        self.ffw1 = all_tensors[f"{p}.feed_forward.w1.weight"]
        self.ffw2 = all_tensors[f"{p}.feed_forward.w2.weight"]
        self.ffw2 = all_tensors[f"{p}.feed_forward.w2.weight"]
        self.attn_norm = RmsNorm(all_tensors[f"{p}.attention_norm.weight"])
        self.ffn_norm = RmsNorm(all_tensors[f"{p}.ffn_norm.weight"])

    def __call__(self, x, index_pos):
        residual = x
        x = self.attn_norm(x)
        attn = self.forward_attn(x, mask, index_pos) 
        x = attn + residual

        residual = x
        x = self.ffn_norm(x)
        w1 = self.ffw1.matmul_t(x)
        w3 = self.ffw3.matmul_t(x)
        mlp = self.ffw2.matmul_t(w1.silu() * w3)

        return mlp + residual

class QuantizedLlama:
    def __init__(self, hparams, all_tensors):
        self.tok_embeddings = all_tensors["tok_embeddings.weight"].dequantize()
        self.norm = RmsNorm(all_tensors["norm.weight"])
        self.output = all_tensors["output.weight"]
        self.layers = []
        for layer_idx in range(hparams["n_layer"]):
            layer = QuantizedLayer(layer_idx, all_tensors)
            self.layers.append(layer)

    def __call__(self, token, index_pos):
        b_size, seq_len = token.shape
        vocab_size, hidden_size = self.tok_embeddings.shape
        token = token.reshape((b_size * seq_len,))
        x = self.tok_embeddings.index_select(token, 0)
        x = x.reshape((b_size, seq_len, hidden_size))
        for layer in self.layers:
            x = layer(x, index_pos)

def main():
    if len(sys.argv) < 2:
        raise ValueError("missing weight file argument")
    filename = sys.argv[1]
    if filename.endswith("gguf"):
        all_tensors = candle.load_gguf(sys.argv[1])
        hparams = None
    else:
        all_tensors, hparams = candle.load_ggml(sys.argv[1])
    print(hparams)
    model = QuantizedLlama(hparams, all_tensors)

    tokens = [1]
    for token_idx in range(1):
        print(tokens)
        last_token = tokens[-1]
        lt = candle.tensor([last_token]).unsqueeze(0)
        logits = model(lt, len(tokens))
        next_token = "TODO"
        tokens.append(next_token)

if __name__ == '__main__':
    main()
