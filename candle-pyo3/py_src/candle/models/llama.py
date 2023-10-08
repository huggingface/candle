from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Union
import candle
from candle import nn
from candle import Tensor
import candle.functional as F
import math


# see https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py
@dataclass
class Config:
    vocab_size = 32000
    hidden_size = 4096
    intermediate_size = 11008
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = None
    hidden_act = "silu"
    max_position_embeddings = 2048
    initializer_range = 0.02
    rms_norm_eps = 1e-6
    use_cache = True
    pad_token_id = None
    bos_token_id = 1
    eos_token_id = 2
    pretraining_tp = 1
    tie_word_embeddings = False
    rope_theta = 10000.0
    rope_scaling = None
    attention_bias = False

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        values = [1.0 / (base ** (i / dim)) for i in range(0, self.dim, 2)]
        self.inv_freq = candle.Tensor(values).to(device)

        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=candle.f32)
        self._non_persistent_buffers_set.add("cos_cached")
        self._non_persistent_buffers_set.add("sin_cached")
        self._non_persistent_buffers_set.add("inv_freq")

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = F.arange(0, self.max_seq_len_cached).to(device, dtype=dtype)

        # original: freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = t.unsqueeze(1).broadcast_mul(self.inv_freq.unsqueeze(0))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = candle.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x: Tensor, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = candle.ones((hidden_size,))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(candle.f32)
        variance = hidden_states.pow(2.0).mean(-1, keepdim=True)
        hidden_states = hidden_states.broadcast_mul(F.rsqrt(variance + self.variance_epsilon))
        return self.weight.broadcast_mul(hidden_states.to(input_dtype))


def rotate_half(x: Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return candle.cat((x2 * -1, x1), dim=-1)


# Copied from transformers.models.gpt_neox.modeling_gpt_neox.apply_rotary_pos_emb
def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, position_ids: Tensor):
    cos = cos[position_ids.squeeze(0)].unsqueeze(0)  # [seq_len, dim] -> [1, seq_len, head_dim]
    sin = sin[position_ids.squeeze(0)].unsqueeze(0)
    q_embed = q.broadcast_mul(cos) + rotate_half(q).broadcast_mul(sin)
    k_embed = k.broadcast_mul(cos) + rotate_half(k).broadcast_mul(sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu if config.hidden_act == "silu" else F.gelu

    def forward(self, x: Tensor):
        if self.config.pretraining_tp > 1:
            raise NotImplementedError("Pretraining TP > 1 not implemented yet")
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: candle.Tensor, n_rep: int) -> candle.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Config):
        super().__init__()

        # for backward compatibility
        if config.num_key_value_heads is None:
            config.num_key_value_heads = config.num_attention_heads

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            raise NotImplementedError("Rope scaling not implemented yet")

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            raise NotImplementedError("Pretraining TP > 1 not implemented yet")
        else:
            query_states: Tensor = self.q_proj(hidden_states)
            key_states: Tensor = self.k_proj(hidden_states)
            value_states: Tensor = self.v_proj(hidden_states)

            query_states = query_states.view((bsz, q_len, self.num_heads, self.head_dim)).transpose(1, 2)
            key_states = key_states.view((bsz, q_len, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
            value_states = value_states.view((bsz, q_len, self.num_key_value_heads, self.head_dim)).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]
            cos, sin = self.rotary_emb.forward(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = candle.cat([past_key_value[0], key_states], dim=2)
                value_states = candle.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights: Tensor = F.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights.broadcast_add(attention_mask)

            # upcast attention to fp32
            attn_weights = F.softmax(attn_weights, dim=-1).to(query_states.dtype)
            attn_output = F.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()

            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            if self.config.pretraining_tp > 1:
                raise NotImplementedError("Pretraining TP > 1 not implemented yet")
            else:
                attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn.forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: Tuple[int], dtype: candle.DType, device: str, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = candle.full((tgt_len, tgt_len), dtype.min, device=device)
    mask_cond = F.arange(0, mask.size(-1)).to(device)
    mask = mask.masked_fill(
        mask_cond.broadcast_as(mask.shape) < (mask_cond + 1).view((mask.size(-1), 1)).broadcast_as(mask.shape), 0
    )
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = candle.cat([candle.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand((bsz, 1, tgt_len, tgt_len + past_key_values_length))


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: candle.Tensor, dtype: candle.DType, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand((bsz, 1, tgt_len, src_len)).to(dtype)

    inverted_mask: Tensor = candle.ones(expanded_mask.shape) - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(candle.u32), dtype.min)


# See https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: candle.Tensor = None,
        attention_mask: Optional[candle.Tensor] = None,
        position_ids: Optional[candle.Tensor] = None,
        past_key_values: Optional[List[candle.Tensor]] = None,
        inputs_embeds: Optional[candle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = F.arange(past_key_values_length, seq_length + past_key_values_length).to(
                device=device, dtype=candle.u32
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = candle.ones(
                (batch_size, seq_length_with_past), dtype=candle.u32, device=inputs_embeds.device
            )
            padding_mask = None
        else:
            if 0 in attention_mask.values():
                padding_mask = attention_mask
            else:
                padding_mask = None

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer.forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        else:
            return {
                "last_hidden_state": hidden_states,
                "past_key_values": next_cache,
                "hidden_states": all_hidden_states,
                "attentions": all_self_attns,
            }


class LlamaForCausalLM(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: candle.Tensor = None,
        attention_mask: Optional[candle.Tensor] = None,
        position_ids: Optional[candle.Tensor] = None,
        past_key_values: Optional[List[candle.Tensor]] = None,
        inputs_embeds: Optional[candle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else False

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            raise NotImplementedError("Pretraining TP > 1 not implemented yet")
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.to(candle.f32)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return output

        return {
            "logits": logits,
            "past_key_values": outputs.past_key_values,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
