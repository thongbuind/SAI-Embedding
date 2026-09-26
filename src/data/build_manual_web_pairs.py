"""Ghép câu hỏi tự sinh với chunk sổ tay web thành dữ liệu contrastive (một nguồn manual_web).

Đầu vào:
- data/embedding/manual_web/chunks.jsonl, clusters.jsonl (prepare_manual_web)
- data/embedding/manual_web/questions/*.jsonl: {"text_key": ..., "queries": [{"q": ..., "type": ...}]}
  Câu hỏi viết cho chunk đại diện của mỗi cụm (các chunk gần trùng giữa các xe).

Đầu ra: data/embedding/mined/manual_web.jsonl với {query, positive, negatives, group, type, model}.
- Mỗi chunk được thay bằng chunk đại diện của cụm nó. Các xe có nhiều mục chỉ khác tên xe, nên sau
  khi quy về đại diện, bản của xe khác trùng text với positive và loss tự mask (không thành
  negative giả), dù các xe chung một nguồn/batch. Mỗi câu hỏi chỉ ghép với đại diện cụm, một lần.
- Hard negative: top BM25 trong sổ tay của xe đại diện, bỏ chunk cùng group và cùng cụm.
- Bỏ câu gần trùng (Jaccard 3-gram ≥ 0,6) với câu dev/eval vf8_manual.
- Bỏ câu mà BM25 không đưa được chunk cùng group vào top BM25_KEEP_TOPK (câu lệch nội dung).
- Bỏ VF 8 (corpus eval) và VF e34 (đã có manual_vfe34): không có cụm nào đại diện bằng hai xe này.

    python -m src.data.build_manual_web_pairs
"""
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

WEB_DIR = Path("data/embedding/manual_web")
OUT_DIR = Path("data/embedding/mined")
HELDOUT = [Path("data/embedding/dev/vf8_manual/queries.jsonl"), Path("data/embedding/eval/vf8_manual/queries.jsonl")]
SKIP_MODELS = {"VF8", "VF e34"}
BM25_KEEP_TOPK = 50
NEG_RANGE = (0, 30)
NUM_NEGATIVES = 3
LEAK_JACCARD = 0.6


def tokens(text: str):
    return re.findall(r"\w+", text.lower())


def trigrams(text: str):
    t = tokens(text)
    return {tuple(t[i:i + 3]) for i in range(len(t) - 2)} or {tuple(t)}


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.tfs = [Counter(tokens(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(w for tf in self.tfs for w in tf)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def rank(self, query):
        q = set(tokens(query))
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            scores.append(s)
        return sorted(range(len(scores)), key=lambda i: -scores[i])


def main():
    rng = random.Random(54)
    chunks = [json.loads(l) for l in open(WEB_DIR / "chunks.jsonl")]
    clusters = [json.loads(l) for l in open(WEB_DIR / "clusters.jsonl")]
    questions = defaultdict(list)
    for path in sorted((WEB_DIR / "questions").glob("*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            questions[r["text_key"]].extend(r["queries"])
    canon = {k: cl["key"] for cl in clusters for k in cl["members"]}
    canon_text = {cl["key"]: cl["text"] for cl in clusters}
    heldout = [trigrams(json.loads(l)["text"]) for p in HELDOUT if p.exists() for l in open(p)]

    def leaks(q):
        g = trigrams(q)
        return any(len(g & h) / len(g | h) >= LEAK_JACCARD for h in heldout)

    by_model = defaultdict(list)
    for c in chunks:
        by_model[c["model"]].append(c)
    bm25 = {}

    rows, dropped = [], Counter()
    for cl in clusters:
        model = cl["rep_model"]
        if model in SKIP_MODELS or cl["key"] not in questions:
            continue
        cs = by_model[model]
        if model not in bm25:
            bm25[model] = BM25([c["text"] for c in cs])
        pos = next(c for c in cs if c["text_key"] == cl["key"])
        seen = set()
        for item in questions[cl["key"]]:
            q = item["q"].strip()
            if not q or q.lower() in seen:
                continue
            seen.add(q.lower())
            if leaks(q):
                dropped["leak"] += 1
                continue
            order = bm25[model].rank(q)
            if pos["group"] not in {cs[i]["group"] for i in order[:BM25_KEEP_TOPK]}:
                dropped["bm25"] += 1
                continue
            negs = []
            for i in order[NEG_RANGE[0]:NEG_RANGE[1]]:
                key = canon.get(cs[i]["text_key"], cs[i]["text_key"])
                text = canon_text.get(key, cs[i]["text"])
                if cs[i]["group"] != pos["group"] and key != cl["key"] and text not in negs:
                    negs.append(text)
            rows.append({"query": q, "positive": cl["text"], "negatives": negs[:NUM_NEGATIVES],
                         "group": pos["group"], "type": item.get("type", ""), "model": model})
    rng.shuffle(rows)
    out = OUT_DIR / "manual_web.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"file": str(out), "rows": len(rows), "clusters": len({r["positive"] for r in rows}),
             "dropped": dict(dropped), "by_model": dict(Counter(r["model"] for r in rows).most_common()),
             "by_type": dict(Counter(r["type"] for r in rows).most_common()),
             "short_negatives": sum(len(r["negatives"]) < NUM_NEGATIVES for r in rows)}
    (WEB_DIR / "pairs_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
