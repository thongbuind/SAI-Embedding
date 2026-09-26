import torch
import torch.nn as nn
from src.model.DecoderBlock import DecoderBlock
from src.model.RotaryPositionalEmbedding import RotaryPositionalEmbedding

class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_kv_heads: int, num_layers: int, ff_dim: int, max_seq_len: int, dropout: float, pad_token_id: int = 0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(d_k, max_seq_len)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, num_kv_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(d_model, eps=1e-6)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        causal = torch.triu(
            torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1
        )
        self.register_buffer("causal_mask", causal, persistent=False)

        self._init_weights()

    def _init_weights(self):
        std = self.d_model ** -0.5
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _build_attn_mask(self, T: int, pad_mask, device, causal: bool = True):
        if not causal:
            # Bidirectional (embedding): chỉ chặn key là PAD, mọi token nhìn thấy nhau.
            pad = torch.zeros(pad_mask.shape[0], 1, 1, T, device=device)
            pad.masked_fill_(~pad_mask[:, None, None, :], float('-inf'))
            return pad
        attn_mask = self.causal_mask[:T, :T][None, None, :, :]
        if pad_mask is not None:
            pad = torch.zeros(pad_mask.shape[0], 1, 1, T, device=device)
            pad.masked_fill_(~pad_mask[:, None, None, :], float('-inf'))
            attn_mask = attn_mask + pad
        return attn_mask

    def forward_features(self, input_ids: torch.Tensor, attention_mask=None, has_padding: bool = True, causal: bool = True) -> torch.Tensor:
        """Giống forward() nhưng DỪNG TRƯỚC lm_head — dùng cho training loss chunked
        (tránh vật lý hóa logits full (B*T, vocab_size)).

        causal=False bật attention hai chiều cho embedding model (LLM2Vec);
        causal=True giữ đúng hành vi của LLM gốc (dùng để đo baseline zero-shot)."""
        pad_mask = attention_mask.bool() if attention_mask is not None \
                   else (input_ids != self.pad_token_id)
        B, T = input_ids.shape
        x = self.embed(input_ids)
        pos = torch.arange(T, device=input_ids.device)
        cos, sin = self.rope.get_cos_sin(pos)

        # Chỉ build mask (và cộng pad-bias) khi batch này thực sự có token PAD.
        # Nếu không có PAD, causal mask thuần == causal+pad mask về mặt toán học
        # (pad-bias toàn số 0) -> bỏ qua an toàn, không đổi kết quả.
        # has_padding được tính sẵn trên CPU trong collate_fn nên không tốn sync GPU ở đây.
        attn_mask = self._build_attn_mask(T, pad_mask, x.device, causal) if has_padding else None

        for block in self.blocks:
            x = block(x, cos, sin, attn_mask, causal)
        return self.norm(x)  # (B, T, d_model) — CHƯA qua lm_head

    def forward(self, input_ids: torch.Tensor, attention_mask=None, has_padding: bool = True) -> torch.Tensor:
        x = self.forward_features(input_ids, attention_mask, has_padding)
        return self.lm_head(x)
