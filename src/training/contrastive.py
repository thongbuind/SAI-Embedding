"""Contrastive fine-tuning cho SAI-Embedding_100M (bước 2 của pipeline).

InfoNCE hai chiều với in-batch negative + hard negative, Matryoshka trên mrl_dims,
GradCache để batch lớn (512+) chạy được trên GPU 24GB. Mỗi batch lấy từ một nguồn.
Chọn checkpoint tốt nhất theo bộ dev (không dùng tập eval).

Chạy: python -m src.training.contrastive [--config config/embedding_train.json] [--resume]
"""
import argparse
import json
import time
from collections import Counter
from contextlib import nullcontext
from functools import partial
from pathlib import Path
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from src.data.embedding_data import (
    EmbeddingTokenizer, PairDataset, SourceBatchSampler, PairCollator, resolve_path,
)
from src.eval.evaluate_embedding import run_suite
from src.model.EmbeddingModel import EmbeddingModel
from src.training.embedding_losses import matryoshka_info_nce, grad_cache_step, direct_step
from src.utils.utils import get_step_lr_lambda, log_progress, select_device, format_time, memory_str

def dev_eval(model, tokenizer, config):
    eval_cfg = config["eval"]
    _, score = run_suite(
        model, tokenizer, config["dev"], eval_cfg["max_query_len"], eval_cfg["max_passage_len"],
        eval_cfg["batch_size"], max_queries=config["dev"].get("max_queries"),
    )
    model.train()
    return score

def main():
    parser = argparse.ArgumentParser(description="Contrastive fine-tuning SAI-Embedding_100M")
    parser.add_argument("--config", default="config/embedding_train.json")
    parser.add_argument("--init", help="Checkpoint khởi tạo (mặc định: contrastive.init_checkpoint)")
    parser.add_argument("--output")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_steps", type=int, help="Dừng sớm sau N step (debug)")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ <output>.ckpt.pt")
    parser.add_argument("--compile", action="store_true", help="torch.compile forward_features (CUDA)")
    args = parser.parse_args()

    with open(resolve_path(args.config), "r", encoding="utf-8") as f:
        config = json.load(f)
    cfg = config["contrastive"]
    init_path = resolve_path(args.init or cfg["init_checkpoint"])
    output_path = resolve_path(args.output or cfg["output"])
    ckpt_path = output_path.with_suffix(".ckpt.pt")
    batch_size = args.batch_size or cfg["batch_size"]
    torch.manual_seed(cfg["seed"])

    device, amp_enabled, amp_dtype = select_device()
    amp_ctx = (lambda: autocast("cuda", dtype=amp_dtype)) if amp_enabled else nullcontext
    scaler = GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    model = EmbeddingModel.from_checkpoint(init_path, defaults=config)
    # Cấu hình pipeline (causal/pooling/prefix/mrl) lấy theo config hiện tại.
    model.causal, model.pooling = config["causal"], config["pooling"]
    model.query_prefix, model.passage_prefix = config["query_prefix"], config["passage_prefix"]
    model.mrl_dims = config["mrl_dims"]
    model.to(device).train()
    if args.compile:
        model.backbone.forward_features = torch.compile(model.backbone.forward_features, dynamic=True)

    tokenizer = EmbeddingTokenizer()
    dataset = PairDataset(cfg["train_files"], cfg.get("max_rows_per_source"), cfg["seed"])
    repeats = {dataset.sources.index(s): r for s, r in cfg.get("source_repeat", {}).items() if s in dataset.sources}
    sampler = SourceBatchSampler(dataset.source_idx, batch_size, cfg["seed"], repeats)
    collator = PairCollator(tokenizer, config["query_prefix"], config["passage_prefix"],
                            cfg["max_query_len"], cfg["max_passage_len"], cfg["num_negatives"],
                            cfg.get("max_passage_len_by_source"))
    # GradCache giữ số token mỗi chunk như ở độ dài passage mặc định, kể cả với nguồn passage dài hơn.
    token_budget = cfg["chunk_size"] * cfg["max_passage_len"]
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collator, num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda", persistent_workers=False,
    )

    steps_per_epoch = len(sampler)
    total_steps = steps_per_epoch * cfg["epochs"]
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        [{"params": [p for p in params if p.ndim >= 2], "weight_decay": cfg["weight_decay"]},
         {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
        lr=cfg["learning_rate"], fused=amp_enabled,
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_step_lr_lambda(cfg["warmup_steps"], total_steps))

    mrl_weights = cfg.get("mrl_weights")
    if mrl_weights is not None and len(mrl_weights) != len(config["mrl_dims"]):
        raise ValueError(f"mrl_weights cần {len(config['mrl_dims'])} phần tử (theo mrl_dims), có {len(mrl_weights)}")
    loss_fn = partial(matryoshka_info_nce, dims=config["mrl_dims"], temperature=cfg["temperature"],
                      symmetric=cfg["symmetric_loss"], weights=mrl_weights)

    start_epoch, global_step, best_score = 0, 0, float("-inf")
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step, best_score = ckpt["global_step"], ckpt["best_score"]
        start_epoch = global_step // steps_per_epoch
        log_progress(f"Resume từ step {global_step} (epoch {start_epoch + 1}), best dev {best_score:.2f}")

    print("╔════════════════════════════════════════════════════════════════════════════════════╗")
    log_progress(f"Contrastive | init={init_path.name} | device={device} | causal={model.causal} pooling={model.pooling}")
    log_progress(f"{len(dataset):,} mẫu {dataset.source_counts()}")
    log_progress(f"batch/epoch theo nguồn {sampler.batches_per_source(dataset.sources)}")
    log_progress(f"batch={batch_size} steps/epoch={steps_per_epoch} total_steps={total_steps} "
                 f"grad_cache={cfg['grad_cache']} chunk={cfg['chunk_size']} τ={cfg['temperature']} mrl={config['mrl_dims']} w={mrl_weights}")

    def save_resume():
        torch.save({
            "global_step": global_step, "best_score": best_score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, ckpt_path)

    def evaluate_and_save():
        nonlocal best_score
        score = dev_eval(model, tokenizer, config)
        if score != score:  # NaN: chưa có bộ dev -> luôn lưu bản mới nhất
            model.save(output_path)
            log_progress(f"step {global_step} | không có dev set, đã lưu {output_path.name}")
        elif score > best_score:
            best_score = score
            model.save(output_path)
            log_progress(f"step {global_step} | dev {score:.2f} ★ best -> {output_path.name}")
        else:
            log_progress(f"step {global_step} | dev {score:.2f} (best {best_score:.2f})")
        save_resume()

    log_every = cfg.get("log_every", 20)
    start, running, running_n, running_gn = time.time(), 0.0, 0, 0.0
    last_log, last_step, window_sources = start, global_step, Counter()
    steps_this_run = 0
    done = global_step >= total_steps
    for epoch in range(start_epoch, cfg["epochs"]):
        if done:
            break
        sampler.set_epoch(epoch, start_batch=global_step - epoch * steps_per_epoch)
        for batch in loader:
            # GradCache tự gọi backward nhiều lần nên nhân scale thủ công; scaler.scale()
            # trên tensor 1 phần tử vừa khởi tạo trạng thái scaler vừa trả về hệ số (chỉ GPU FP16).
            scale = scaler.scale(torch.ones((), device=device)).item() if scaler.is_enabled() else 1.0
            batch_loss_fn = partial(loss_fn, q_keys=batch["q_keys"].to(device), p_keys=batch["p_keys"].to(device),
                                    p_groups=batch["p_groups"].to(device))
            if cfg["grad_cache"]:
                loss = grad_cache_step(model, batch, device, amp_ctx, batch_loss_fn, cfg["chunk_size"], scale,
                                       token_budget)
            else:
                loss = direct_step(model, batch, device, amp_ctx, batch_loss_fn, scale)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            steps_this_run += 1

            running += loss.item()
            running_gn += float(grad_norm)
            running_n += 1
            window_sources[batch["source"]] += 1
            now = time.time()
            if global_step % log_every == 0 or global_step == total_steps:
                elapsed, window = now - start, now - last_log
                eta = elapsed / steps_this_run * (total_steps - global_step)
                n_steps = global_step - last_step
                log_progress(f"ep {epoch + 1} step {global_step}/{total_steps} ({100 * global_step / total_steps:.1f}%) "
                             f"| loss {running / running_n:.4f} | grad {running_gn / running_n:.3f} "
                             f"| lr {scheduler.get_last_lr()[0]:.2e}")
                log_progress(f"  {n_steps / window:.2f} step/s | {n_steps * batch_size / window:,.0f} mẫu/s "
                             f"| đã chạy {format_time(elapsed)} | ETA {format_time(eta)}{memory_str(device)}")
                log_progress("  nguồn: " + ", ".join(f"{s} {c}" for s, c in window_sources.most_common()))
                running, running_n, running_gn = 0.0, 0, 0.0
                last_log, last_step, window_sources = now, global_step, Counter()
            if global_step % cfg["eval_every"] == 0:
                evaluate_and_save()
                last_log = time.time()  # không tính thời gian eval vào tốc độ train
            if global_step >= total_steps:
                done = True
                break

    if global_step % cfg["eval_every"] != 0:
        evaluate_and_save()
    log_progress(f"Xong sau {format_time(time.time() - start)}. Model tốt nhất: {output_path}")
    print("╚════════════════════════════════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
