from dataclasses import dataclass
from typing import Optional
from candle.nn import Module, Embedding, LayerNorm, Linear, ModuleList
from candle import Tensor
import candle
import candle.functional as F
from typing import Tuple, Optional


@dataclass
class Config:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    use_cache: bool = True
    classifier_dropout: Optional[float] = None
    model_type: Optional[str] = "bert"


class BertSelfAttention(Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        all_head_size = int(config.num_attention_heads * self.attention_head_size)
        hidden_size = config.hidden_size
        self.query = Linear(hidden_size, all_head_size)
        self.key = Linear(hidden_size, all_head_size)
        self.value = Linear(hidden_size, all_head_size)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.reshape(new_x_shape).transpose(1, 2)
        return x.contiguous()

    def forward(self, hidden_states: Tensor, attention_mask=None) -> Tensor:
        query = self.query.forward(hidden_states)
        key = self.key.forward(hidden_states)
        value = self.value.forward(hidden_states)

        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        attention_scores = query.matmul(key.t())
        attention_scores = attention_scores / float(self.attention_head_size) ** 0.5
        if attention_mask is not None:
            b_size, _, _, last_dim = attention_scores.shape
            attention_scores = attention_scores.broadcast_add(attention_mask.reshape((b_size, 1, 1, last_dim)))
        attention_probs = F.softmax(attention_scores, dim=-1)

        context_layer = attention_probs.matmul(value)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.flatten_from(-2)
        return context_layer


class BertSelfOutput(Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense.forward(hidden_states)
        return self.LayerNorm.forward(hidden_states + input_tensor)


class BertAttention(Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states: Tensor, attention_mask: None) -> Tensor:
        self_outputs = self.self.forward(hidden_states, attention_mask=attention_mask)
        attention_output = self.output.forward(self_outputs, hidden_states)
        return attention_output


class BertIntermediate(Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size)
        self.act = F.gelu if config.hidden_act == "gelu" else F.relu

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense.forward(hidden_states)
        return self.act(hidden_states)


class BertOutput(Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense.forward(hidden_states)
        return self.LayerNorm.forward(hidden_states + input_tensor)


class BertLayer(Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: Tensor, attention_mask=None) -> Tensor:
        attention_output = self.attention.forward(hidden_states, attention_mask=attention_mask)
        # TODO: Support cross-attention?
        # https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L523
        # TODO: Support something similar to `apply_chunking_to_forward`?
        intermediate_output = self.intermediate.forward(attention_output)
        layer_output = self.output.forward(intermediate_output, attention_output)
        return layer_output


class BertEncoder(Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.layer = ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layer.append(BertLayer(config))

    def forward(self, hidden_states: Tensor, attention_mask=None) -> Tensor:
        for l in self.layer:
            hidden_states = l.forward(hidden_states, attention_mask=attention_mask)
        return hidden_states


class BertEmbeddings(Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.position_ids = candle.Tensor(list(range(config.max_position_embeddings))).reshape(
            (1, config.max_position_embeddings)
        )

    def forward(self, input_ids: Tensor, token_type_ids: Tensor) -> Tensor:
        (_batch_size, seq_len) = input_ids.shape
        input_embeddings = self.word_embeddings.forward(input_ids)
        token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)
        embeddings: Tensor = input_embeddings + token_type_embeddings

        position_ids = list(range(seq_len))
        position_ids = Tensor(position_ids).to_dtype(input_ids.dtype).to_device(input_ids.device)

        embeddings = embeddings.broadcast_add(self.position_embeddings.forward(position_ids))
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class BertPooler(Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.activation = F.tanh

    def forward(self, hidden_states: Tensor) -> Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense.forward(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def masked_fill(on_false: float, mask: Tensor, on_true: float):
    shape = mask.shape
    on_true = candle.tensor(on_true).broadcast_as(shape)
    on_false = candle.tensor(on_false).broadcast_as(shape)
    return mask.where_cond(on_true, on_false)


# https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L874
class BertModel(Module):
    def __init__(self, config: Config, add_pooling_layer=True) -> None:
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

    def forward(
        self, input_ids: Tensor, token_type_ids: Tensor, attention_mask=None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if attention_mask is not None:
            # Replace 0s with -inf, and 1s with 0s.
            attention_mask = masked_fill(float("-inf"), attention_mask, 1.0)
        embeddings = self.embeddings.forward(input_ids, token_type_ids)
        encoder_out = self.encoder.forward(embeddings, attention_mask=attention_mask)
        pooled_output = self.pooler(encoder_out) if self.pooler is not None else None
        return encoder_out, pooled_output
