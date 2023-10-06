import candle
from typing import Dict, Tuple, Any
from candle import Tensor, QTensor, utils, nn
from candle.nn import Module, ModuleList


def masked_fill(on_false: Tensor, mask: Tensor, on_true: Tensor):
    shape = mask.shape
    on_true = candle.tensor(on_true).broadcast_as(shape)
    return mask.where_cond(on_true, on_false)


def precompute_freqs_cis(hparams: Dict[str, Any], freq_base: float, max_seq_len: int):
    head_dim = hparams["n_embd"] // hparams["n_head"]
    theta = [1.0 / freq_base ** (i / head_dim) for i in range(0, head_dim, 2)]
    theta = candle.tensor(theta)
    idx_theta = [float(i) for i in range(max_seq_len)]
    idx_theta = candle.tensor(idx_theta).reshape((max_seq_len, 1))
    m = idx_theta.matmul(theta.unsqueeze(0))
    return (m.cos(), m.sin())


class RmsNorm(Module):
    def __init__(self, qtensor: QTensor):
        super().__init__()
        self.weight = qtensor.dequantize()

    def forward(self, x: Tensor) -> Tensor:
        b_size, seq_len, hidden_size = x.shape
        norm_x = x.sqr().sum_keepdim(2) / hidden_size
        x_normed = x.broadcast_div((norm_x + 1e-5).sqrt())
        return x_normed.broadcast_mul(self.weight)


class QuantizedLayer(Module):
    def __init__(
        self,
        layer_idx: int,
        hparams: Dict[str, Any],
        all_tensors: Dict[str, QTensor],
        cos_sin: Tuple[Tensor, Tensor],
    ):
        super().__init__()
        p = f"layers.{layer_idx}"
        self.attention_wq = all_tensors[f"{p}.attention.wq.weight"]
        self.attention_wk = all_tensors[f"{p}.attention.wk.weight"]
        self.attention_wv = all_tensors[f"{p}.attention.wv.weight"]
        self.attention_wo = all_tensors[f"{p}.attention.wo.weight"]
        self.ffw1 = all_tensors[f"{p}.feed_forward.w1.weight"]
        self.ffw2 = all_tensors[f"{p}.feed_forward.w2.weight"]
        self.ffw3 = all_tensors[f"{p}.feed_forward.w3.weight"]
        self.attn_norm = RmsNorm(all_tensors[f"{p}.attention_norm.weight"])
        self.ffn_norm = RmsNorm(all_tensors[f"{p}.ffn_norm.weight"])

        self.n_head = hparams["n_head"]
        self.n_kv_head = self.n_head
        self.head_dim = hparams["n_embd"] // self.n_head

        self.kv_cache = None
        self.cos = cos_sin[0]
        self.sin = cos_sin[1]
        self._non_persistent_buffers_set.add("cos")
        self._non_persistent_buffers_set.add("sin")

    def forward(self, x: Tensor, mask: Tensor, index_pos: int) -> Tensor:
        residual = x
        x = self.attn_norm(x)
        attn = self.forward_attn(x, mask, index_pos)
        x = attn + residual

        residual = x
        x = self.ffn_norm(x)
        w1 = self.ffw1.matmul_t(x)
        w3 = self.ffw3.matmul_t(x)
        mlp = self.ffw2.matmul_t(nn.silu(w1) * w3)

        return mlp + residual

    def forward_attn(self, x: Tensor, mask: Tensor, index_pos: int):
        b_size, seq_len, n_embd = x.shape
        q = self.attention_wq.matmul_t(x)
        k = self.attention_wk.matmul_t(x)
        v = self.attention_wv.matmul_t(x)

        q = q.reshape((b_size, seq_len, self.n_head, self.head_dim)).transpose(1, 2)
        k = k.reshape((b_size, seq_len, self.n_kv_head, self.head_dim)).transpose(1, 2)
        v = v.reshape((b_size, seq_len, self.n_kv_head, self.head_dim)).transpose(1, 2)

        q = self.apply_rotary_emb(q, index_pos)
        k = self.apply_rotary_emb(k, index_pos)

        if self.kv_cache is not None and index_pos > 0:
            prev_k, prev_v = self.kv_cache
            k = candle.cat([prev_k, k], 2).contiguous()
            v = candle.cat([prev_v, v], 2).contiguous()

        self.kv_cache = (k, v)

        # TODO: maybe repeat k/v here if we start supporting MQA.

        att = q.matmul(k.t()) / self.head_dim**0.5
        mask = mask.broadcast_as(att.shape)
        att = masked_fill(att, mask, float("-inf"))
        att = nn.softmax(att, -1)
        y = att.matmul(v.contiguous())
        y = y.transpose(1, 2).reshape((b_size, seq_len, n_embd))
        return self.attention_wo.matmul_t(y)

    def apply_rotary_emb(self, x: Tensor, index_pos: int):
        b_size, n_head, seq_len, n_embd = x.shape
        cos = self.cos.narrow(0, index_pos, seq_len).reshape((seq_len, n_embd // 2, 1))
        sin = self.sin.narrow(0, index_pos, seq_len).reshape((seq_len, n_embd // 2, 1))
        x = x.reshape((b_size, n_head, seq_len, n_embd // 2, 2))
        x0 = x.narrow(-1, 0, 1)
        x1 = x.narrow(-1, 1, 1)
        y0 = x0.broadcast_mul(cos) - x1.broadcast_mul(sin)
        y1 = x0.broadcast_mul(sin) + x1.broadcast_mul(cos)
        rope = candle.cat([y0, y1], -1)
        return rope.flatten_from(-2)


class QuantizedLlama(Module):
    def __init__(self, hparams: Dict[str, Any], all_tensors: Dict[str, QTensor]):
        super().__init__()
        self.tok_embeddings = all_tensors["tok_embeddings.weight"].dequantize()
        self.norm = RmsNorm(all_tensors["norm.weight"])
        self.output = all_tensors["output.weight"]
        self.layers = ModuleList()
        rope_freq = hparams.get("rope_freq", 10000.0)
        cos_sin = precompute_freqs_cis(hparams, rope_freq, hparams["context_length"])
        for layer_idx in range(hparams["n_layer"]):
            layer = QuantizedLayer(layer_idx, hparams, all_tensors, cos_sin)
            self.layers.append(layer)

    def forward(self, token: Tensor, index_pos: int) -> Tensor:
        b_size, seq_len = token.shape
        vocab_size, hidden_size = self.tok_embeddings.shape
        token = token.reshape((b_size * seq_len,))
        x = self.tok_embeddings.index_select(token, 0)
        x = x.reshape((b_size, seq_len, hidden_size))

        mask = [int(j > i) for j in range(seq_len) for i in range(seq_len)]
        mask = candle.tensor(mask).reshape((seq_len, seq_len))

        for layer in self.layers:
            x = layer(x, mask, index_pos)
        x = self.norm(x)
        x = x.narrow(1, -1, 1).squeeze(1)
        x = self.output.matmul_t(x)
        return x
