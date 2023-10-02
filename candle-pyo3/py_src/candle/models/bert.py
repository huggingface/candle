from dataclasses import dataclass
from typing import Optional
from candle.nn import Module

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



class BertLayerNorm(Module):
    pass

class BertSelfAttention(Module):
    pass

class BertEncoder(Module):
    pass

class BertPooler(Module):
    pass

class BertEmbeddings(Module):
    pass


class BertModel(Module):
    def __init__(self, config:Config) -> None:
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()
    
