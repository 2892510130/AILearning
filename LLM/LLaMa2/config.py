from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"
    def __init__(
            self,
            dim: int = 768,          # Model Dimension
            n_layers: int = 12,      # Transformer Layers
            n_heads: int = 16,       # Attention Header
            n_kv_heads: int = 8,     # key-value head number
            vocab_size: int = 6144,  # Vocab Size
            hidden_dim: int = None,  # Hidden Dimension
            multiple_of: int = 64,   # check hidden size and head size relation
            norm_eps: float = 1e-5,  # normal layer eps
            max_seq_len: int = 512,  # max sequence length
            dropout: float = 0.0,    # dropout probability
            flash_attn: bool = True, # whether use Flash Attention
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)