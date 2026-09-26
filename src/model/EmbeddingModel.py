import json
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.TransformerModel import TransformerModel

project_root = Path(__file__).resolve().parent.parent.parent
config_dir = project_root / "config"

def build_backbone(model_size: str) -> TransformerModel:
    with open(config_dir / "base.json", "r") as f:
        config = json.load(f)
    with open(config_dir / f"{model_size}.json", "r") as f:
        config.update(json.load(f))
    return TransformerModel(
        config["vocab_size"], config["d_model"], config["num_heads"], config["num_kv_heads"],
        config["num_layers"], config["ff_dim"], config["max_seq_len"], config["dropout"],
    )

def _extract_state_dict(obj):
    """Nhận state_dict thuần (pretrained_*.pt), checkpoint resume ({model_state_dict})
    hoặc checkpoint embedding ({embedding_config, state_dict})."""
    config = {}
    if isinstance(obj, dict) and "embedding_config" in obj:
        config, state_dict = obj["embedding_config"], obj["state_dict"]
    elif isinstance(obj, dict) and "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
    else:
        state_dict = obj
    # Bỏ prefix do torch.compile thêm vào (nếu có).
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return config, state_dict

class EmbeddingModel(nn.Module):
    """SAI backbone -> pooling -> vector câu.

    - causal=False: attention hai chiều (cần adapt bằng MNTP trước khi dùng).
    - pooling="mean": trung bình các token không phải PAD; "last": token cuối
      (chỉ hợp lý khi causal=True, dùng để so sánh baseline).
    - Không có projection head: vector 768 chiều, cắt được theo Matryoshka (mrl_dims).
    """
    def __init__(self, backbone: TransformerModel, model_size: str, causal: bool = False,
                 pooling: str = "mean", query_prefix: str = "", passage_prefix: str = "",
                 mrl_dims=None):
        super().__init__()
        assert pooling in ("mean", "last"), pooling
        self.backbone = backbone
        self.model_size = model_size
        self.causal = causal
        self.pooling = pooling
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.mrl_dims = list(mrl_dims or [backbone.d_model])

    @property
    def embedding_config(self):
        return {
            "model_size": self.model_size, "causal": self.causal, "pooling": self.pooling,
            "query_prefix": self.query_prefix, "passage_prefix": self.passage_prefix,
            "mrl_dims": self.mrl_dims,
        }

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu", defaults=None, **overrides):
        """Thứ tự ưu tiên cấu hình: overrides > cấu hình lưu trong checkpoint > defaults.

        Checkpoint LM thuần (pretrained_100M.pt, mntp_100M.pt) không lưu cấu hình,
        nên truyền defaults lấy từ config/embedding.json."""
        obj = torch.load(path, map_location=map_location, weights_only=False)
        saved_config, state_dict = _extract_state_dict(obj)
        config = {"model_size": "100M", "causal": False, "pooling": "mean",
                  "query_prefix": "", "passage_prefix": "", "mrl_dims": None}
        config.update({k: v for k, v in (defaults or {}).items() if k in config})
        config.update(saved_config)
        config.update({k: v for k, v in overrides.items() if v is not None})
        backbone = build_backbone(config["model_size"])
        backbone.load_state_dict(state_dict)
        return cls(backbone, **config)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embedding_config": self.embedding_config,
                    "state_dict": self.backbone.state_dict()}, path)

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(hidden.dtype)
        if self.pooling == "mean":
            summed = (hidden.float() * mask.float().unsqueeze(-1)).sum(dim=1)
            return summed / mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        last = attention_mask.float().sum(dim=1).long() - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last].float()

    def forward(self, input_ids, attention_mask, has_padding: bool = True, normalize: bool = True):
        hidden = self.backbone.forward_features(
            input_ids, attention_mask, has_padding=has_padding, causal=self.causal,
        )
        pooled = self.pool(hidden, attention_mask)
        return F.normalize(pooled, dim=-1) if normalize else pooled

    @torch.no_grad()
    def encode(self, texts, tokenizer, is_query: bool = True, max_len: int = 512,
               batch_size: int = 128, dim: int = None, prefix: str = None, show_progress: bool = False):
        """Encode list text -> tensor (N, dim) float32 đã L2-normalize, trên CPU.

        Text được sắp theo độ dài để giảm padding rồi trả về đúng thứ tự ban đầu.
        """
        if prefix is None:
            prefix = self.query_prefix if is_query else self.passage_prefix
        device = next(self.parameters()).device
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        was_training = self.training
        self.eval()

        tokenized = tokenizer.encode(list(texts), prefix, max_len)
        order = sorted(range(len(tokenized)), key=lambda i: -len(tokenized[i]))
        dim = dim or self.backbone.d_model
        out = torch.empty(len(tokenized), dim, dtype=torch.float32)
        for step, start in enumerate(range(0, len(order), batch_size)):
            idx = order[start:start + batch_size]
            input_ids, attention_mask, has_padding = tokenizer.pad([tokenized[i] for i in idx])
            with amp:
                emb = self(input_ids.to(device), attention_mask.to(device), has_padding, normalize=False)
            out[idx] = F.normalize(emb[:, :dim].float(), dim=-1).cpu()
            if show_progress and step % 50 == 0:
                print(f"  encode {start + len(idx):,}/{len(order):,}", end="\r")
        if show_progress:
            print()
        self.train(was_training)
        return out
