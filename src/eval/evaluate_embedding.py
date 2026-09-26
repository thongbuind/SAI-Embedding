import argparse
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from src.data.embedding_data import EmbeddingTokenizer, iter_jsonl, resolve_path
from src.utils.utils import log_progress, select_device

project_root = Path(__file__).resolve().parent.parent.parent

def load_retrieval_task(task_dir: Path):
    """Định dạng kiểu BEIR: corpus.jsonl {_id,title,text,group?}, queries.jsonl {_id,text},
    qrels.tsv (query-id, corpus-id, score) có dòng header.
    Nếu corpus có trường group (vd. mục tài liệu), qrels ghi theo group thay vì theo văn bản."""
    corpus, groups = {}, {}
    for row in iter_jsonl(task_dir / "corpus.jsonl"):
        title, text = row.get("title") or "", row.get("text") or ""
        corpus[str(row["_id"])] = f"{title}\n{text}".strip() if title else text
        if row.get("group") is not None:
            groups[str(row["_id"])] = str(row["group"])
    queries = {str(r["_id"]): r["text"] for r in iter_jsonl(task_dir / "queries.jsonl")}
    qrels = {}
    with open(task_dir / "qrels.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for qid, did, score in reader:
            if float(score) > 0:
                qrels.setdefault(qid, {})[did] = float(score)
    return corpus, queries, qrels, groups

def _truncate(emb: torch.Tensor, dim):
    return emb if not dim or dim >= emb.size(1) else F.normalize(emb[:, :dim], dim=-1)

def retrieval_metrics(ranked_lists, qids, qrels):
    """ranked_lists[i]: danh sách id (văn bản hoặc group) đã xếp hạng cho qids[i]."""
    ndcg, mrr, r10, r100 = [], [], [], []
    for ranked, qid in zip(ranked_lists, qids):
        rel = qrels[qid]
        dcg = sum(rel.get(d, 0.0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        ndcg.append(dcg / idcg if idcg > 0 else 0.0)
        rr = 0.0
        for i, d in enumerate(ranked[:10]):
            if d in rel:
                rr = 1.0 / (i + 1)
                break
        mrr.append(rr)
        r10.append(len(set(ranked[:10]) & rel.keys()) / len(rel))
        r100.append(len(set(ranked[:100]) & rel.keys()) / len(rel))
    return {
        "ndcg@10": 100 * float(np.mean(ndcg)), "mrr@10": 100 * float(np.mean(mrr)),
        "recall@10": 100 * float(np.mean(r10)), "recall@100": 100 * float(np.mean(r100)),
    }

def evaluate_retrieval(encoder, tokenizer, task_dir, max_query_len=64, max_passage_len=512,
                       batch_size=128, dims=(None,), max_queries=None, seed=0):
    corpus, queries, qrels, groups = load_retrieval_task(Path(task_dir))
    qids = [q for q in queries if q in qrels]
    if max_queries and len(qids) > max_queries:
        rng = np.random.default_rng(seed)
        qids = sorted(rng.choice(qids, size=max_queries, replace=False).tolist())
    doc_ids = list(corpus.keys())

    doc_emb = encoder.encode([corpus[d] for d in doc_ids], tokenizer, is_query=False,
                             max_len=max_passage_len, batch_size=batch_size)
    q_emb = encoder.encode([queries[q] for q in qids], tokenizer, is_query=True,
                           max_len=max_query_len, batch_size=batch_size)

    results = {}
    # Chấm theo group: lấy sâu hơn để sau khi gộp các đoạn cùng group vẫn còn đủ 100 group.
    k = min(1000 if groups else 100, len(doc_ids))
    for dim in dims:
        d_emb, qe = _truncate(doc_emb, dim), _truncate(q_emb, dim)
        top_idx = []
        for start in range(0, len(qids), 256):
            top_idx.append((qe[start:start + 256] @ d_emb.T).topk(k, dim=1).indices)
        ranked = [[doc_ids[j] for j in row] for row in torch.cat(top_idx).tolist()]
        if groups:
            # Hạng của một group = hạng đoạn cao nhất của nó (chấm theo mục).
            ranked = [list(dict.fromkeys(groups[d] for d in row)) for row in ranked]
        metrics = retrieval_metrics(ranked, qids, qrels)
        results[f"dim{dim or doc_emb.size(1)}"] = metrics
    results["num_queries"], results["num_docs"] = len(qids), len(doc_ids)
    return results

def evaluate_sts(encoder, tokenizer, path, max_len=128, batch_size=128, dims=(None,)):
    """STS là task đối xứng: cả hai câu dùng query prefix."""
    rows = list(iter_jsonl(path))
    s1 = encoder.encode([r["sentence1"] for r in rows], tokenizer, is_query=True,
                        max_len=max_len, batch_size=batch_size)
    s2 = encoder.encode([r["sentence2"] for r in rows], tokenizer, is_query=True,
                        max_len=max_len, batch_size=batch_size)
    gold = [float(r["score"]) for r in rows]
    results = {}
    for dim in dims:
        cos = (_truncate(s1, dim) * _truncate(s2, dim)).sum(dim=1).tolist()
        results[f"dim{dim or s1.size(1)}"] = {"spearman": 100 * float(spearmanr(cos, gold).statistic)}
    results["num_pairs"] = len(rows)
    return results

def run_suite(encoder, tokenizer, suite_cfg: dict, max_query_len=64, max_passage_len=512,
              batch_size=128, dims=(None,), max_queries=None, verbose=True):
    """Chạy các task retrieval + STS có trong suite_cfg; task thiếu file thì bỏ qua.
    Trả về (results, main_score) với main_score = trung bình nDCG@10 / Spearman ở chiều đầy đủ."""
    results, main = {}, []
    full_dim_key = None
    for name, task_dir in suite_cfg.get("retrieval_tasks", {}).items():
        task_dir = resolve_path(task_dir)
        if not (task_dir / "qrels.tsv").exists():
            if verbose:
                log_progress(f"[eval] Bỏ qua {name}: chưa có {task_dir}")
            continue
        start = time.time()
        res = evaluate_retrieval(encoder, tokenizer, task_dir, max_query_len, max_passage_len,
                                 batch_size, dims, max_queries)
        full_dim_key = full_dim_key or next(k for k in res if k.startswith("dim"))
        results[name] = res
        main.append(res[full_dim_key]["ndcg@10"])
        if verbose:
            log_progress(f"[eval] {name:<14} nDCG@10={res[full_dim_key]['ndcg@10']:.2f} "
                         f"R@100={res[full_dim_key]['recall@100']:.2f} "
                         f"({res['num_queries']} q, {res['num_docs']} docs, {time.time() - start:.0f}s)")
    for name, path in suite_cfg.get("sts_tasks", {}).items():
        path = resolve_path(path)
        if not path.exists():
            if verbose:
                log_progress(f"[eval] Bỏ qua {name}: chưa có {path}")
            continue
        res = evaluate_sts(encoder, tokenizer, path, max_query_len * 2, batch_size, dims)
        key = next(k for k in res if k.startswith("dim"))
        results[name] = res
        main.append(res[key]["spearman"])
        if verbose:
            log_progress(f"[eval] {name:<14} Spearman={res[key]['spearman']:.2f} ({res['num_pairs']} cặp)")
    main_score = float(np.mean(main)) if main else float("nan")
    results["main_score"] = main_score
    return results, main_score

class HFEncoder:
    """Bọc model Hugging Face (vd. intfloat/multilingual-e5-small) cùng interface encode()
    để so sánh baseline trên đúng bộ eval. Không lowercase, dùng tokenizer riêng của model.
    segment=True: tách từ bằng pyvi trước khi tokenize (model gốc PhoBERT, vd. bkai vietnamese-bi-encoder)."""
    def __init__(self, name, device, query_prefix="", passage_prefix="", pooling="mean", segment=False):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.device, self.pooling = device, pooling
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.segment = None
        if segment:
            from pyvi.ViTokenizer import tokenize
            self.segment = tokenize

    @torch.no_grad()
    def encode(self, texts, tokenizer=None, is_query=True, max_len=512, batch_size=128, dim=None):
        prefix = self.query_prefix if is_query else self.passage_prefix
        if self.segment:
            texts = [self.segment(t) for t in texts]
        texts = [prefix + t for t in texts]
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        out = None
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                           max_length=max_len, return_tensors="pt").to(self.device)
            hidden = self.model(**enc).last_hidden_state.float()
            if self.pooling == "cls":
                emb = hidden[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).float()
                emb = (hidden * m).sum(1) / m.sum(1)
            emb = F.normalize(emb, dim=-1).cpu()
            if out is None:
                out = torch.empty(len(texts), emb.size(1))
            out[idx] = emb
        return out

def main():
    parser = argparse.ArgumentParser(description="Đánh giá SAI-Embedding_100M trên các task tiếng Việt")
    parser.add_argument("--config", default="config/embedding.json")
    parser.add_argument("--checkpoint", help="Checkpoint embedding, hoặc checkpoint LM (pretrained_100M.pt) để đo zero-shot")
    parser.add_argument("--hf_model", help="Tên model HF để đo baseline, vd. intfloat/multilingual-e5-small")
    parser.add_argument("--hf_query_prefix", default="query: ")
    parser.add_argument("--hf_passage_prefix", default="passage: ")
    parser.add_argument("--hf_pooling", default="mean", choices=["mean", "cls"])
    parser.add_argument("--hf_segment", action="store_true", help="Tách từ bằng pyvi (model dựa trên PhoBERT)")
    parser.add_argument("--causal", action="store_true", help="Ghi đè: attention causal (baseline zero-shot)")
    parser.add_argument("--bidirectional", action="store_true", help="Ghi đè: attention hai chiều")
    parser.add_argument("--pooling", choices=["mean", "last"], help="Ghi đè cách pooling")
    parser.add_argument("--split", default="eval", choices=["eval", "dev"])
    parser.add_argument("--dims", type=int, nargs="*", help="Các chiều Matryoshka cần đo, vd. 768 256 128")
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--tasks", nargs="*", help="Chỉ chạy các task này, vd. --tasks zalo_legal")
    parser.add_argument("--name", help="Tên file kết quả trong results/")
    args = parser.parse_args()

    with open(resolve_path(args.config), "r", encoding="utf-8") as f:
        config = json.load(f)
    eval_cfg = config["eval"]
    suite_cfg = config[args.split]
    if args.tasks:
        suite_cfg = {**suite_cfg, **{kind: {k: v for k, v in suite_cfg.get(kind, {}).items() if k in args.tasks}
                                     for kind in ("retrieval_tasks", "sts_tasks")}}
    device, _, _ = select_device()
    tokenizer = EmbeddingTokenizer()

    if args.hf_model:
        encoder = HFEncoder(args.hf_model, device, args.hf_query_prefix, args.hf_passage_prefix, args.hf_pooling,
                             args.hf_segment)
        name = args.name or args.hf_model.replace("/", "__")
    else:
        from src.model.EmbeddingModel import EmbeddingModel
        causal = True if args.causal else (False if args.bidirectional else None)
        encoder = EmbeddingModel.from_checkpoint(
            resolve_path(args.checkpoint), defaults=config, causal=causal, pooling=args.pooling,
        )
        encoder.to(device).eval()
        name = args.name or f"{Path(args.checkpoint).stem}_{'causal' if encoder.causal else 'bidir'}_{encoder.pooling}"

    log_progress(f"Đánh giá {name} trên split '{args.split}' ({device})")
    results, main_score = run_suite(
        encoder, tokenizer, suite_cfg, eval_cfg["max_query_len"], eval_cfg["max_passage_len"],
        eval_cfg["batch_size"], args.dims or (None,), args.max_queries or suite_cfg.get("max_queries"),
    )
    log_progress(f"Main score (trung bình nDCG@10 / Spearman): {main_score:.2f}")
    out = project_root / "results" / f"{name}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log_progress(f"Đã lưu {out.relative_to(project_root)}")

if __name__ == "__main__":
    main()
