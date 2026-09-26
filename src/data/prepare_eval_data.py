"""Tải + chuẩn hoá các bộ đánh giá công khai cho SAI-Embedding_100M (định dạng BEIR).

Đầu ra (data/embedding/eval/):
  zalo_legal/, viquad/, tvpl_legal/, msmarco_nano/   corpus.jsonl, queries.jsonl, qrels.tsv
  sts_vn.jsonl                                       {sentence1, sentence2, score}

Chỉ dùng split test (hoặc dev với nano-MSMARCO), không đụng tới split train.

Chạy: python -m src.data.prepare_eval_data [--tasks ...]
"""
import argparse
import hashlib
import random
from pathlib import Path
import pandas as pd
from src.data.embedding_data import iter_jsonl, write_jsonl, normalize_text, project_root
from src.utils.utils import log_progress

OUT_DIR = project_root / "data" / "embedding" / "eval"

def fetch(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download
    log_progress(f"[tải] {repo_id}/{filename}")
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

def read_parquet(repo_id, filename, columns=None) -> pd.DataFrame:
    return pd.read_parquet(fetch(repo_id, filename), columns=columns)

def doc_id(text: str) -> str:
    return hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()[:16]

def clean(text) -> str:
    # Một số nguồn lưu xuống dòng dạng chữ "\\n" -> đưa về khoảng trắng.
    return " ".join(str(text).replace("\\n", " ").split()) if text is not None else ""

def write_beir(task_dir: Path, corpus: dict, queries: dict, qrels: dict):
    """corpus {id: (title, text)}, queries {id: text}, qrels {qid: {did: score}}."""
    task_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(task_dir / "corpus.jsonl",
                ({"_id": d, "title": t, "text": x}
                 for d, (t, x) in corpus.items()))
    write_jsonl(task_dir / "queries.jsonl", ({"_id": q, "text": x} for q, x in queries.items()))
    with open(task_dir / "qrels.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, docs in qrels.items():
            for did, score in docs.items():
                f.write(f"{qid}\t{did}\t{score}\n")
    log_progress(f"[beir] {task_dir.relative_to(project_root)}: {len(queries)} query, {len(corpus)} docs")

def prepare_zalo_legal(args):
    repo = "GreenNode/zalo-ai-legal-text-retrieval-vn"
    corpus = {}
    for row in iter_jsonl(fetch(repo, "corpus.jsonl")):
        did = str(row.get("_id", row.get("id")))
        corpus[did] = (clean(row.get("title")), clean(row.get("text")))
    queries = {str(r.get("_id", r.get("id"))): clean(r["text"]) for r in iter_jsonl(fetch(repo, "queries.jsonl"))}
    qrels = {}
    for r in iter_jsonl(fetch(repo, "qrels/test.jsonl")):
        q, d = str(r["query-id"]), str(r["corpus-id"])
        if float(r["score"]) > 0 and d in corpus and q in queries:
            qrels.setdefault(q, {})[d] = 1
    write_beir(OUT_DIR / "zalo_legal", corpus, {q: queries[q] for q in qrels}, qrels)

def prepare_viquad(args):
    df = read_parquet("taidng/UIT-ViQuAD2.0", "data/test-00000-of-00001.parquet", ["context", "question", "is_impossible"])
    df = df[~df["is_impossible"].astype(bool)]
    corpus, queries, qrels = {}, {}, {}
    for ctx, question in zip(df["context"], df["question"]):
        ctx, question = clean(ctx), clean(question)
        did, qid = doc_id(ctx), doc_id(question)
        corpus[did] = ("", ctx)
        queries[qid] = question
        qrels.setdefault(qid, {})[did] = 1
    write_beir(OUT_DIR / "viquad", corpus, queries, qrels)

def prepare_sts(args):
    df = read_parquet("GreenNode/stsbenchmark-sts-vn", "data/test-00000-of-00001.parquet", ["sentence1", "sentence2", "score"])
    out = OUT_DIR / "sts_vn.jsonl"
    n = write_jsonl(out, ({"sentence1": clean(a), "sentence2": clean(b), "score": float(s)}
                          for a, b, s in zip(df["sentence1"], df["sentence2"], df["score"])))
    log_progress(f"[sts] {out.relative_to(project_root)}: {n} cặp")

def _prepare_beir_parquet(repo, out_name, corpus_file, queries_file, qrels_file, max_queries=None, seed=54):
    corpus_df = read_parquet(repo, corpus_file, ["id", "title", "text"])
    corpus = {str(i): (clean(t), clean(x)) for i, t, x in zip(corpus_df["id"], corpus_df["title"], corpus_df["text"])}
    queries_df = read_parquet(repo, queries_file, ["id", "text"])
    queries = {str(i): clean(t) for i, t in zip(queries_df["id"], queries_df["text"])}
    qrels_df = read_parquet(repo, qrels_file)
    qrels = {}
    for q, d, s in zip(qrels_df["query-id"], qrels_df["corpus-id"], qrels_df["score"]):
        if float(s) > 0 and str(q) in queries and str(d) in corpus:
            qrels.setdefault(str(q), {})[str(d)] = 1
    if max_queries and len(qrels) > max_queries:
        keep = random.Random(seed).sample(sorted(qrels), max_queries)
        qrels = {q: qrels[q] for q in sorted(keep)}
    write_beir(OUT_DIR / out_name, corpus, {q: queries[q] for q in qrels}, qrels)

def prepare_tvpl(args):
    _prepare_beir_parquet("GreenNode/TVPL-Retrieval-VN", "tvpl_legal",
                          "corpus/test-00000-of-00001.parquet", "queries/test-00000-of-00001.parquet",
                          "qrels/test-00000-of-00001.parquet", max_queries=3000, seed=args.seed)

def prepare_msmarco_nano(args):
    _prepare_beir_parquet("GreenNode/nano-msmarco-vn", "msmarco_nano",
                          "corpus/dev-00000-of-00001.parquet", "queries/dev-00000-of-00001.parquet",
                          "qrels/dev-00000-of-00001.parquet", seed=args.seed)

PREPARERS = {
    "zalo_legal": prepare_zalo_legal, "viquad": prepare_viquad, "sts": prepare_sts,
    "tvpl": prepare_tvpl, "msmarco_nano": prepare_msmarco_nano,
}

def main():
    parser = argparse.ArgumentParser(description="Tải các bộ đánh giá công khai cho SAI-Embedding_100M")
    parser.add_argument("--tasks", nargs="*", default=list(PREPARERS), help=f"Mặc định: {' '.join(PREPARERS)}")
    parser.add_argument("--seed", type=int, default=54)
    args = parser.parse_args()
    for task in args.tasks:
        log_progress(f"── {task} ──")
        PREPARERS[task](args)
    log_progress("Xong.")

if __name__ == "__main__":
    main()
