"""MNTP (masked next-token prediction), bước 1 của LLM2Vec.

Bật attention hai chiều cho backbone decoder-only rồi dạy model dùng ngữ cảnh
hai phía: mask ~20% token, dự đoán token bị mask ở vị trí i từ hidden state ở
vị trí i-1 bằng lm_head cũ. Khớp với cách model đã học (next-token), nên chỉ cần
vài nghìn step để model hết "ngơ ngác" khi nhìn thấy token tương lai.

Chạy: python -m src.training.mntp [--config config/embedding_train.json]
"""
import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from src.data.embedding_data import EmbeddingTokenizer, iter_jsonl, normalize_text, resolve_path
from src.model.EmbeddingModel import build_backbone, _extract_state_dict
from src.utils.chunked_loss import chunked_lm_loss
from src.utils.utils import get_step_lr_lambda, log_progress, select_device, format_time, memory_str

# Tokenizer không có [MASK]; dùng [UNK] (id 0) như LLM2Vec dùng "_" cho Llama.
# An toàn vì attention_mask luôn được truyền tường minh (không suy ra từ id 0).
MASK_ID = 0
FIRST_REGULAR_ID = 5  # 0..4 là [UNK] [BOS] [EOS] <|im_start|> <|im_end|>
MIN_CHUNK_TOKENS = 32

def iter_chunks(path, tokenizer, seq_len, skip_docs=0, max_docs=None):
    """Đọc {"text"} -> các đoạn [BOS] + tối đa seq_len-1 token, không nối qua ranh giới văn bản."""
    for doc_idx, row in enumerate(iter_jsonl(path)):
        if doc_idx < skip_docs:
            continue
        if max_docs is not None and doc_idx >= max_docs:
            return
        text = row.get("text")
        if not text:
            continue
        ids = tokenizer.sp.encode(normalize_text(text), out_type=int)
        for start in range(0, len(ids), seq_len - 1):
            piece = ids[start:start + seq_len - 1]
            if len(piece) >= MIN_CHUNK_TOKENS:
                yield [tokenizer.bos_id] + piece

def train_batches(path, tokenizer, seq_len, batch_size, skip_docs, buffer_size=20_000, seed=54):
    """Stream vô hạn (lặp lại file khi hết) với shuffle buffer."""
    rng = random.Random(seed)
    buffer = []
    while True:
        produced = False
        for chunk in iter_chunks(path, tokenizer, seq_len, skip_docs=skip_docs):
            produced = True
            buffer.append(chunk)
            if len(buffer) >= buffer_size:
                rng.shuffle(buffer)
                while len(buffer) >= buffer_size // 2:
                    yield tokenizer.pad([buffer.pop() for _ in range(batch_size)])
        if not produced:
            raise ValueError(f"Không có văn bản train nào trong {path}")
        rng.shuffle(buffer)
        while len(buffer) >= batch_size:
            yield tokenizer.pad([buffer.pop() for _ in range(batch_size)])

def mask_tokens(input_ids, attention_mask, mask_ratio, vocab_size, generator=None):
    """BERT-style 80/10/10 trên các token thật (bỏ [BOS] ở vị trí 0 vì không có vị trí i-1)."""
    device = input_ids.device
    candidates = attention_mask.bool().clone()
    candidates[:, 0] = False
    probs = torch.rand(input_ids.shape, generator=generator, device="cpu").to(device)
    masked = candidates & (probs < mask_ratio)
    action = torch.rand(input_ids.shape, generator=generator, device="cpu").to(device)
    random_ids = torch.randint(FIRST_REGULAR_ID, vocab_size, input_ids.shape, generator=generator).to(device)
    corrupted = input_ids.clone()
    corrupted[masked & (action < 0.8)] = MASK_ID
    swap = masked & (action >= 0.8) & (action < 0.9)
    corrupted[swap] = random_ids[swap]
    return corrupted, masked

def mntp_loss(backbone, features_fn, input_ids, attention_mask, has_padding, mask_ratio, generator=None):
    corrupted, masked = mask_tokens(input_ids, attention_mask, mask_ratio,
                                    backbone.lm_head.weight.size(0), generator)
    hidden = features_fn(corrupted, attention_mask, has_padding, causal=False)
    # Token bị mask ở vị trí i được dự đoán từ hidden ở vị trí i-1.
    select = masked[:, 1:]
    h = hidden[:, :-1][select]
    targets = input_ids[:, 1:][select]
    n = targets.numel()
    if n == 0:
        return hidden.sum() * 0.0, 0
    loss_sum = chunked_lm_loss(h, backbone.lm_head.weight, targets,
                               torch.ones(n, device=h.device, dtype=torch.float32))
    return loss_sum / n, n

@torch.no_grad()
def evaluate(backbone, features_fn, val_batches, device, amp_ctx, mask_ratio):
    backbone.eval()
    generator = torch.Generator().manual_seed(0)  # cùng mask mỗi lần eval -> so sánh được
    total, count = 0.0, 0
    for input_ids, attention_mask, has_padding in val_batches:
        with amp_ctx():
            loss, n = mntp_loss(backbone, features_fn, input_ids.to(device), attention_mask.to(device),
                                has_padding, mask_ratio, generator)
        total += loss.item() * n
        count += n
    backbone.train()
    return total / max(count, 1)

def main():
    parser = argparse.ArgumentParser(description="MNTP: adapt SAI sang attention hai chiều")
    parser.add_argument("--config", default="config/embedding_train.json")
    parser.add_argument("--init", help="Checkpoint LM gốc (mặc định: base_checkpoint trong config)")
    parser.add_argument("--data", help="JSONL {text} (mặc định: mntp.data trong config)")
    parser.add_argument("--output")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--val_docs", type=int, help="Số văn bản đầu file dành cho validation")
    parser.add_argument("--compile", action="store_true", help="torch.compile forward_features (CUDA)")
    args = parser.parse_args()

    with open(resolve_path(args.config), "r", encoding="utf-8") as f:
        config = json.load(f)
    cfg = config["mntp"]
    init_path = resolve_path(args.init or config["base_checkpoint"])
    data_path = resolve_path(args.data or cfg["data"])
    output_path = resolve_path(args.output or cfg["output"])
    steps = args.steps or cfg["steps"]
    batch_size = args.batch_size or cfg["batch_size"]
    seq_len = args.seq_len or cfg["seq_len"]
    mask_ratio = cfg["mask_ratio"]

    device, amp_enabled, amp_dtype = select_device()
    amp_ctx = (lambda: autocast("cuda", dtype=amp_dtype)) if amp_enabled else nullcontext
    scaler = GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    backbone = build_backbone(config["model_size"])
    _, state_dict = _extract_state_dict(torch.load(init_path, map_location="cpu", weights_only=False))
    backbone.load_state_dict(state_dict)
    backbone.to(device).train()
    features_fn = torch.compile(backbone.forward_features, dynamic=True) if args.compile else backbone.forward_features

    tokenizer = EmbeddingTokenizer()
    val_docs = args.val_docs if args.val_docs is not None else cfg["val_docs"]
    val_chunks = list(iter_chunks(data_path, tokenizer, seq_len, max_docs=val_docs))[:batch_size * 20]
    val_batches = [tokenizer.pad(val_chunks[i:i + batch_size]) for i in range(0, len(val_chunks), batch_size)]
    batches = train_batches(data_path, tokenizer, seq_len, batch_size, skip_docs=val_docs)

    decay = [p for p in backbone.parameters() if p.ndim >= 2]
    no_decay = [p for p in backbone.parameters() if p.ndim < 2]
    optimizer = optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["learning_rate"], fused=amp_enabled,
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_step_lr_lambda(cfg["warmup_steps"], steps))

    print("╔════════════════════════════════════════════════════════════════════════════════════╗")
    log_progress(f"MNTP | init={init_path.name} | data={data_path.name} | device={device}")
    log_progress(f"steps={steps} batch={batch_size} seq_len={seq_len} mask_ratio={mask_ratio} val_batches={len(val_batches)}")
    val_loss = evaluate(backbone, features_fn, val_batches, device, amp_ctx, mask_ratio)
    log_progress(f"step 0 | val MNTP loss {val_loss:.4f} (ppl {math.exp(min(val_loss, 20)):.1f})")

    log_every = cfg.get("log_every", 50)
    start, running, running_n, running_gn, tokens = time.time(), 0.0, 0, 0.0, 0
    last_log, last_step = start, 0
    for step in range(1, steps + 1):
        input_ids, attention_mask, has_padding = next(batches)
        tokens += int(attention_mask.sum())  # đếm trên CPU: tránh sync và phép giảm int64 trên MPS
        input_ids, attention_mask = input_ids.to(device, non_blocking=True), attention_mask.to(device, non_blocking=True)
        with amp_ctx():
            loss, _ = mntp_loss(backbone, features_fn, input_ids, attention_mask, has_padding, mask_ratio)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        running += loss.item()
        running_gn += float(grad_norm)
        running_n += 1
        now = time.time()
        if step % log_every == 0 or step == steps:
            elapsed, window = now - start, now - last_log
            log_progress(f"step {step}/{steps} ({100 * step / steps:.1f}%) | loss {running / running_n:.4f} "
                         f"| grad {running_gn / running_n:.3f} | lr {scheduler.get_last_lr()[0]:.2e}")
            log_progress(f"  {tokens / window:,.0f} tok/s | {(step - last_step) / window:.2f} step/s "
                         f"| đã chạy {format_time(elapsed)} | ETA {format_time(elapsed / step * (steps - step))}"
                         f"{memory_str(device)}")
            running, running_n, running_gn, tokens = 0.0, 0, 0.0, 0
            last_log, last_step = now, step
        if step % cfg["eval_every"] == 0 or step == steps:
            val_loss = evaluate(backbone, features_fn, val_batches, device, amp_ctx, mask_ratio)
            log_progress(f"step {step} | val MNTP loss {val_loss:.4f} (ppl {math.exp(min(val_loss, 20)):.1f})")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Lưu dạng state_dict thuần như pretrained_*.pt để dùng lại được ở mọi nơi.
            torch.save(backbone.state_dict(), output_path)
            last_log = time.time()  # không tính thời gian eval vào tốc độ train

    log_progress(f"Xong MNTP sau {format_time(time.time() - start)} -> {output_path}")
    print("╚════════════════════════════════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
