"""Benchmark SAI-Embedding_100M với một embedding model cùng cỡ trên Hugging Face,
rồi xuất kết quả thành một dashboard HTML duy nhất (không cần internet để mở).

Chạy (cần GPU hoặc máy đủ RAM; bộ eval phải có sẵn, hoặc thêm --prepare_data để tải):
    python scripts/benchmark_dashboard.py
    python scripts/benchmark_dashboard.py --baselines intfloat/multilingual-e5-small --max_queries 500
    python scripts/benchmark_dashboard.py --render_only      # chỉ vẽ lại dashboard từ kết quả đã lưu

Kết quả từng model được cache ở results/benchmark/<tên>.json; chạy lại sẽ dùng cache
trừ khi có --rerun. Dashboard: results/benchmark_dashboard.html
"""
import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from src.data.embedding_data import EmbeddingTokenizer, iter_jsonl, resolve_path
from src.eval.evaluate_embedding import HFEncoder, run_suite
from src.utils.utils import log_progress, select_device

cache_dir = project_root / "results" / "benchmark"

# Baseline cùng cỡ (~100M tham số, hỗ trợ tiếng Việt): prefix + pooling theo model card.
KNOWN_BASELINES = {
    "intfloat/multilingual-e5-small": {"query_prefix": "query: ", "passage_prefix": "passage: ", "pooling": "mean"},
    "intfloat/multilingual-e5-base": {"query_prefix": "query: ", "passage_prefix": "passage: ", "pooling": "mean"},
    # bge-m3 (dense): CLS pooling, không prefix. ~568M tham số, 1024 chiều — lớn hơn SAI ~5,6 lần.
    "BAAI/bge-m3": {"query_prefix": "", "passage_prefix": "", "pooling": "cls"},
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {"query_prefix": "", "passage_prefix": "", "pooling": "mean"},
    # PhoBERT-base fine-tune cho retrieval (~135M tham số, 768 chiều): input phải tách từ (pyvi),
    # position embedding của PhoBERT chỉ đủ 256 token.
    "bkai-foundation-models/vietnamese-bi-encoder": {"query_prefix": "", "passage_prefix": "", "pooling": "mean",
                                                     "segment": True, "max_len": 256},
}

TASK_INFO = {
    "zalo_legal": {"label": "Zalo Legal", "domain": "Luật"},
    "tvpl_legal": {"label": "TVPL", "domain": "Luật"},
    "viquad": {"label": "ViQuAD", "domain": "Wikipedia"},
    "msmarco_nano": {"label": "nano-MSMARCO", "domain": "Web"},
    "vf8_manual": {"label": "VF8 manual", "domain": "Sổ tay"},
    "sts_vn": {"label": "STS-B-vi", "domain": "Tương đồng câu"},
}


def count_params(module):
    return sum(p.numel() for p in module.parameters())  # tham số dùng chung (tied) chỉ tính 1 lần


def sample_passages(suite_cfg, n):
    """Lấy n văn bản từ corpus retrieval đầu tiên có sẵn để đo tốc độ encode."""
    for task_dir in suite_cfg.get("retrieval_tasks", {}).values():
        path = resolve_path(task_dir) / "corpus.jsonl"
        if not path.exists():
            continue
        texts = []
        for row in iter_jsonl(path):
            texts.append(f"{row.get('title') or ''}\n{row.get('text') or ''}".strip())
            if len(texts) >= n:
                break
        return texts
    return []


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_speed(encoder, tokenizer, device, passages, max_len, batch_size):
    """Throughput encode passage (văn bản/giây) và độ trễ encode 1 query (ms)."""
    if not passages:
        return {}
    encoder.encode(passages[:batch_size], tokenizer, is_query=False, max_len=max_len, batch_size=batch_size)  # warmup
    sync(device)
    start = time.perf_counter()
    encoder.encode(passages, tokenizer, is_query=False, max_len=max_len, batch_size=batch_size)
    sync(device)
    docs_per_s = len(passages) / (time.perf_counter() - start)

    query = ["người đi bộ vượt đèn đỏ bị phạt bao nhiêu tiền"]
    for _ in range(3):
        encoder.encode(query, tokenizer, is_query=True, max_len=64, batch_size=1)
    times = []
    for _ in range(20):
        sync(device)
        start = time.perf_counter()
        encoder.encode(query, tokenizer, is_query=True, max_len=64, batch_size=1)
        sync(device)
        times.append((time.perf_counter() - start) * 1000)
    return {"docs_per_s": docs_per_s, "query_latency_ms": float(np.median(times)), "num_speed_docs": len(passages)}


def benchmark_model(spec, args, config, suite_cfg, device, tokenizer, passages):
    eval_cfg = config["eval"]
    if spec["kind"] == "sai":
        from src.model.EmbeddingModel import EmbeddingModel
        encoder = EmbeddingModel.from_checkpoint(resolve_path(spec["path"]), defaults=config).to(device).eval()
        params, dim = count_params(encoder.backbone), encoder.backbone.d_model
        dims = [d for d in args.dims if d <= dim] or [dim]
        max_len = 512
    else:
        opts = KNOWN_BASELINES.get(spec["path"], {"query_prefix": args.hf_query_prefix,
                                                  "passage_prefix": args.hf_passage_prefix,
                                                  "pooling": args.hf_pooling})
        encoder = HFEncoder(spec["path"], device, opts["query_prefix"], opts["passage_prefix"], opts["pooling"],
                            opts.get("segment", False))
        params, dim = count_params(encoder.model), encoder.model.config.hidden_size
        dims = [dim]  # baseline không train Matryoshka, chỉ đo chiều đầy đủ
        max_len = min(opts.get("max_len", 512), getattr(encoder.tok, "model_max_length", 512) or 512)

    log_progress(f"== {spec['label']}: {params / 1e6:.1f}M tham số, {dim} chiều ({device}) ==")
    start = time.time()
    results, main_score = run_suite(
        encoder, tokenizer, suite_cfg, eval_cfg["max_query_len"], min(eval_cfg["max_passage_len"], max_len),
        eval_cfg["batch_size"], dims, args.max_queries or suite_cfg.get("max_queries"),
    )
    eval_seconds = time.time() - start
    speed = measure_speed(encoder, tokenizer, device, passages, min(256, max_len), args.speed_batch_size)
    log_progress(f"Main score {main_score:.2f} | {speed.get('docs_per_s', 0):.0f} văn bản/s")

    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return {
        "name": spec["name"], "label": spec["label"], "source": spec["path"], "kind": spec["kind"],
        "params": params, "dim": dim, "dims": dims, "main_score": main_score,
        "results": results, "speed": speed, "eval_seconds": eval_seconds,
        "device": str(device), "split": args.split, "created": datetime.now().isoformat(timespec="seconds"),
    }


def build_task_list(models):
    tasks = []
    for m in models:
        for key, res in m["results"].items():
            if not isinstance(res, dict) or key in [t["key"] for t in tasks]:
                continue
            info = TASK_INFO.get(key, {"label": key, "domain": ""})
            is_sts = "num_pairs" in res
            tasks.append({"key": key, "label": info["label"], "domain": info["domain"],
                          "type": "sts" if is_sts else "retrieval",
                          "size": f"{res['num_pairs']:,} cặp" if is_sts
                                  else f"{res['num_queries']:,} query / {res['num_docs']:,} văn bản"})
    return tasks


def render_dashboard(models, out_path):
    data = {"generated": datetime.now().strftime("%d/%m/%Y %H:%M"), "models": models, "tasks": build_task_list(models)}
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log_progress(f"Đã lưu dashboard: {out_path.relative_to(project_root) if out_path.is_relative_to(project_root) else out_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SAI-Embedding_100M với model cùng cỡ + dashboard HTML")
    parser.add_argument("--config", default="config/embedding.json")
    parser.add_argument("--checkpoint", default="model/SAI-Embedding_100M.pt")
    parser.add_argument("--name", default="SAI-Embedding_100M")
    parser.add_argument("--baselines", nargs="*", default=["intfloat/multilingual-e5-small", "BAAI/bge-m3",
                                                      "bkai-foundation-models/vietnamese-bi-encoder"],
                        help="Model HF để so sánh (tối đa 3 để biểu đồ dễ đọc)")
    parser.add_argument("--hf_query_prefix", default="query: ")
    parser.add_argument("--hf_passage_prefix", default="passage: ")
    parser.add_argument("--hf_pooling", default="mean", choices=["mean", "cls"])
    parser.add_argument("--split", default="eval", choices=["eval", "dev"])
    parser.add_argument("--tasks", nargs="*", help="Chỉ chạy các task này")
    parser.add_argument("--dims", type=int, nargs="*", default=[768, 512, 256, 128, 64],
                        help="Các chiều Matryoshka đo cho SAI (chiều đầu tiên dùng cho main score)")
    parser.add_argument("--max_queries", type=int, default=None, help="Giới hạn số query mỗi task cho nhanh")
    parser.add_argument("--speed_docs", type=int, default=1024, help="Số văn bản dùng để đo tốc độ")
    parser.add_argument("--speed_batch_size", type=int, default=64)
    parser.add_argument("--prepare_data", action="store_true", help="Tải bộ eval trước khi chạy")
    parser.add_argument("--rerun", action="store_true", help="Bỏ qua cache, đo lại mọi model")
    parser.add_argument("--render_only", action="store_true", help="Chỉ vẽ dashboard từ cache")
    parser.add_argument("--out", default="results/benchmark_dashboard.html")
    args = parser.parse_args()

    specs = [{"name": "sai", "label": args.name, "path": args.checkpoint, "kind": "sai"}]
    specs += [{"name": b.replace("/", "__"), "label": b.split("/")[-1], "path": b, "kind": "hf"} for b in args.baselines]
    if len(specs) > 4:
        parser.error("Tối đa 3 baseline để dashboard dễ so sánh")

    models = []
    if not args.render_only:
        with open(resolve_path(args.config), "r", encoding="utf-8") as f:
            config = json.load(f)
        suite_cfg = config[args.split]
        if args.tasks:
            suite_cfg = {**suite_cfg, **{kind: {k: v for k, v in suite_cfg.get(kind, {}).items() if k in args.tasks}
                                         for kind in ("retrieval_tasks", "sts_tasks")}}
        if args.prepare_data:
            from src.data.prepare_eval_data import main as prepare_eval_data
            sys.argv = [sys.argv[0]]
            prepare_eval_data()
        device, _, _ = select_device()
        tokenizer = EmbeddingTokenizer()
        passages = sample_passages(suite_cfg, args.speed_docs)
        cache_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        cache = cache_dir / f"{spec['name']}_{args.split}.json"
        if cache.exists() and (args.render_only or not args.rerun):
            log_progress(f"Dùng kết quả đã lưu: {cache.relative_to(project_root)}")
            models.append(json.loads(cache.read_text(encoding="utf-8")))
            continue
        if args.render_only:
            log_progress(f"Bỏ qua {spec['label']}: chưa có {cache.relative_to(project_root)}")
            continue
        if spec["kind"] == "sai" and not resolve_path(spec["path"]).exists():
            log_progress(f"Bỏ qua {spec['label']}: không thấy checkpoint {spec['path']}")
            continue
        try:
            result = benchmark_model(spec, args, config, suite_cfg, device, tokenizer, passages)
        except Exception as e:  # một model lỗi không làm mất dashboard của các model còn lại
            log_progress(f"LỖI khi đo {spec['label']}: {type(e).__name__}: {str(e)[:120]}")
            continue
        cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        models.append(result)

    if not models:
        log_progress("Không có kết quả nào để vẽ dashboard")
        return
    render_dashboard(models, resolve_path(args.out))


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Embedding Benchmark</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --border: rgba(11,11,11,0.10);
  --text: #0b0b0b; --text-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --wash: rgba(11,11,11,0.04);
  --s1: #4a3aa7; --s2: #eb6834; --s3: #1baf7a; --s4: #c98a00;
  --up: #006300; --down: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --border: rgba(255,255,255,0.10);
    --text: #ffffff; --text-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --wash: rgba(255,255,255,0.05);
    --s1: #9085e9; --s2: #d95926; --s3: #199e70; --s4: #e0a82e;
    --up: #0ca30c; --down: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --border: rgba(255,255,255,0.10);
  --text: #ffffff; --text-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --wash: rgba(255,255,255,0.05);
  --s1: #9085e9; --s2: #d95926; --s3: #199e70; --s4: #e0a82e;
  --up: #0ca30c; --down: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 32px 16px 64px; }
header { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; flex-wrap: wrap; margin-bottom: 24px; }
h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 16px; margin: 0; }
.sub { color: var(--text-2); margin: 0; }
.note { color: var(--muted); font-size: 12px; margin: 4px 0 0; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
.grid { display: grid; gap: 16px; }
.kpis { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-bottom: 16px; }
.two { grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); margin-top: 16px; }
.kpi .who { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; color: var(--text-2); font-weight: 600; }
.kpi .big { font-size: 36px; font-weight: 650; margin: 8px 0 2px; letter-spacing: -0.02em; }
.kpi .meta { color: var(--muted); font-size: 12px; }
.kpi .meta b { color: var(--text-2); font-weight: 600; }
.swatch { width: 10px; height: 10px; border-radius: 3px; display: inline-block; flex: none; }
.badge { font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 99px; background: var(--wash); color: var(--text-2); }
.delta-up { color: var(--up); } .delta-down { color: var(--down); }
.head { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
.seg { display: inline-flex; background: var(--wash); border-radius: 8px; padding: 3px; gap: 2px; flex-wrap: wrap; }
.seg button { border: 0; background: transparent; color: var(--text-2); font: inherit; font-size: 12px; font-weight: 600;
  padding: 5px 10px; border-radius: 6px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--surface); color: var(--text); box-shadow: 0 1px 2px rgba(0,0,0,.12); }
.legend { display: flex; gap: 16px; flex-wrap: wrap; color: var(--text-2); font-size: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--text-2); font-size: 12px; font-family: inherit; }
svg .muted { fill: var(--muted); font-size: 11px; }
svg .val { fill: var(--text); font-size: 12px; font-weight: 600; font-variant-numeric: tabular-nums; }
.table-wrap { overflow-x: auto; margin-top: 16px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 13px; }
th, td { padding: 8px 10px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; font-size: 12px; }
td.best { font-weight: 700; }
.task-name { font-weight: 600; } .task-meta { color: var(--muted); font-size: 11px; display: block; }
#tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--text); border: 1px solid var(--border);
  box-shadow: 0 4px 16px rgba(0,0,0,.14); border-radius: 8px; padding: 8px 10px; font-size: 12px; opacity: 0; transition: opacity .1s; z-index: 10; max-width: 260px; }
#tip .row { display: flex; align-items: center; gap: 6px; }
.hit { fill: transparent; cursor: default; }
.hit:hover { fill: var(--wash); }
.empty { color: var(--muted); padding: 24px 0; text-align: center; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Benchmark embedding tiếng Việt</h1>
      <p class="sub" id="subtitle"></p>
      <p class="note">Main score = trung bình nDCG@10 (retrieval) và Spearman (STS) ở chiều đầy đủ. Thang 0–100, cao hơn là tốt hơn.</p>
    </div>
    <div class="legend" id="legend"></div>
  </header>

  <section class="grid kpis" id="kpis"></section>

  <section class="card">
    <div class="head">
      <div><h2>Điểm theo từng task</h2><p class="note" id="metric-note"></p></div>
      <div class="seg" id="metric-seg" role="group" aria-label="Chọn metric"></div>
    </div>
    <div id="task-chart"></div>
    <div class="table-wrap"><table id="task-table"></table></div>
  </section>

  <section class="grid two">
    <div class="card">
      <div class="head"><div><h2>Chênh lệch so với baseline</h2><p class="note" id="delta-note"></p></div>
        <div class="seg" id="delta-seg" role="group" aria-label="Chọn baseline"></div></div>
      <div id="delta-chart"></div>
    </div>
    <div class="card">
      <div class="head"><div><h2>Matryoshka: điểm khi cắt chiều</h2><p class="note">Main score của SAI ở từng số chiều; đường nét đứt là baseline ở chiều đầy đủ.</p></div></div>
      <div id="mrl-chart"></div>
    </div>
  </section>

  <section class="card" style="margin-top:16px">
    <div class="head"><div><h2>Kích thước và tốc độ</h2><p class="note" id="speed-note"></p></div></div>
    <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(240px,1fr))" id="eff-charts"></div>
  </section>
</div>
<div id="tip" role="tooltip"></div>

<script>
const DATA = __DATA__;
const models = DATA.models, tasks = DATA.tasks;
const COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];
const color = i => COLORS[i % COLORS.length];
const METRICS = [
  { key: "ndcg@10", label: "nDCG@10" }, { key: "mrr@10", label: "MRR@10" },
  { key: "recall@10", label: "Recall@10" }, { key: "recall@100", label: "Recall@100" },
];
const fullKey = m => "dim" + m.dim;
const fmt = (v, d = 1) => v == null || isNaN(v) ? "–" : v.toFixed(d);
const fmtInt = v => v == null ? "–" : Math.round(v).toLocaleString("vi-VN");
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
};
function score(m, task, metric, dimKey) {
  const r = m.results[task.key]; if (!r) return null;
  const d = r[dimKey || fullKey(m)]; if (!d) return null;
  return task.type === "sts" ? d.spearman : d[metric];
}

// Tooltip
const tip = document.getElementById("tip");
function showTip(evt, html) {
  tip.innerHTML = html; tip.style.opacity = 1;
  const x = Math.min(evt.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.min(evt.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.style.opacity = 0; };
const tipRow = (i, name, val) => `<div class="row"><span class="swatch" style="background:${color(i)}"></span>${esc(name)}: <b>${val}</b></div>`;

// Header
document.getElementById("subtitle").textContent =
  `${models.map(m => m.label).join(" vs ")} · split ${models[0].split} · ${models[0].device} · tạo lúc ${DATA.generated}`;
document.getElementById("legend").innerHTML = models.map((m, i) =>
  `<span><span class="swatch" style="background:${color(i)}"></span>${esc(m.label)}</span>`).join("");

// KPI cards
const best = Math.max(...models.map(m => m.main_score));
const ref = models[0];
document.getElementById("kpis").innerHTML = models.map((m, i) => {
  const delta = i === 0 ? null : ref.main_score - m.main_score;
  const wins = i === 0 ? null : tasks.filter(t => {
    const a = score(ref, t, "ndcg@10"), b = score(m, t, "ndcg@10");
    return a != null && b != null && a > b;
  }).length;
  return `<div class="card kpi">
    <div class="who"><span class="swatch" style="background:${color(i)}"></span>${esc(m.label)}
      ${m.main_score === best && models.length > 1 ? '<span class="badge">★ Cao nhất</span>' : ""}</div>
    <div class="big">${fmt(m.main_score, 2)}</div>
    <div class="meta">Main score</div>
    <div class="meta" style="margin-top:8px"><b>${fmt(m.params / 1e6)}M</b> tham số · <b>${m.dim}</b> chiều
      ${m.speed && m.speed.docs_per_s ? ` · <b>${fmtInt(m.speed.docs_per_s)}</b> văn bản/s` : ""}</div>
    ${delta == null ? "" : `<div class="meta" style="margin-top:4px">SAI ${delta >= 0 ? "hơn" : "kém"}
      <b class="${delta >= 0 ? "delta-up" : "delta-down"}">${delta >= 0 ? "▲ +" : "▼ "}${fmt(Math.abs(delta), 2)}</b> điểm · thắng ${wins}/${tasks.length} task</div>`}
  </div>`;
}).join("");

// Grouped horizontal bars per task
let metric = "ndcg@10";
const seg = document.getElementById("metric-seg");
seg.innerHTML = METRICS.map(mt => `<button type="button" data-k="${mt.key}" aria-pressed="${mt.key === metric}">${mt.label}</button>`).join("");
seg.addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  metric = b.dataset.k;
  seg.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x.dataset.k === metric));
  drawTasks();
});

function drawTasks() {
  const label = METRICS.find(x => x.key === metric).label;
  document.getElementById("metric-note").textContent = `Retrieval: ${label} · STS: Spearman · chiều đầy đủ của mỗi model`;
  const box = document.getElementById("task-chart"); box.innerHTML = "";
  const W = Math.max(300, box.clientWidth), labelW = W < 520 ? 110 : 170, right = 56, barH = 16, gap = 2, groupGap = 18;
  const groupH = models.length * (barH + gap) - gap;
  const H = tasks.length * (groupH + groupGap) + 24;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Điểm theo task" }, box);
  const x = v => labelW + (W - labelW - right) * Math.max(0, v) / 100;
  for (const t of [0, 25, 50, 75, 100]) {
    el("line", { x1: x(t), x2: x(t), y1: 0, y2: H - 20, stroke: t ? "var(--grid)" : "var(--axis)", "stroke-width": 1 }, svg);
    el("text", { x: x(t), y: H - 4, "text-anchor": "middle", class: "muted" }, svg).textContent = t;
  }
  tasks.forEach((t, ti) => {
    const y0 = ti * (groupH + groupGap) + 4;
    const name = el("text", { x: 0, y: y0 + groupH / 2 - 2, "dominant-baseline": "middle" }, svg);
    name.textContent = t.label; name.style.fontWeight = 600; name.style.fill = "var(--text)";
    el("text", { x: 0, y: y0 + groupH / 2 + 13, "dominant-baseline": "middle", class: "muted" }, svg)
      .textContent = t.type === "sts" ? "Spearman" : t.domain;
    const vals = models.map(m => score(m, t, metric));
    const top = Math.max(...vals.filter(v => v != null));
    models.forEach((m, i) => {
      const v = vals[i], y = y0 + i * (barH + gap);
      if (v == null) {
        el("text", { x: labelW + 4, y: y + barH / 2, "dominant-baseline": "middle", class: "muted" }, svg).textContent = "chưa đo";
        return;
      }
      const w = Math.max(2, x(v) - labelW);
      el("path", { d: `M${labelW},${y}h${w - 4}a4,4 0 0 1 4,4v${barH - 8}a4,4 0 0 1 -4,4h${-(w - 4)}z`, fill: color(i) }, svg);
      const tx = el("text", { x: x(v) + 6, y: y + barH / 2, "dominant-baseline": "middle", class: "val" }, svg);
      tx.textContent = fmt(v) + (v === top && models.length > 1 ? " ★" : "");
    });
    const hit = el("rect", { x: 0, y: y0 - groupGap / 2, width: W, height: groupH + groupGap, class: "hit" }, svg);
    hit.addEventListener("mousemove", e => showTip(e,
      `<b>${esc(t.label)}</b> <span style="color:var(--muted)">${esc(t.size)}</span>` +
      models.map((m, i) => tipRow(i, m.label, fmt(vals[i], 2))).join("")));
    hit.addEventListener("mouseleave", hideTip);
  });
  drawTable();
}

function drawTable() {
  const cols = [...METRICS.map(x => x.key)];
  let h = `<thead><tr><th>Task</th><th>Model</th>${METRICS.map(x => `<th>${x.label}</th>`).join("")}<th>Spearman</th></tr></thead><tbody>`;
  tasks.forEach(t => {
    const bestOf = k => Math.max(...models.map(m => {
      const r = m.results[t.key]; const d = r && r[fullKey(m)]; return d && d[k] != null ? d[k] : -Infinity;
    }));
    models.forEach((m, i) => {
      const r = m.results[t.key], d = r && r[fullKey(m)];
      const cell = k => {
        const v = d ? d[k] : null;
        return `<td class="${v != null && v === bestOf(k) && models.length > 1 ? "best" : ""}">${fmt(v, 2)}</td>`;
      };
      h += `<tr>${i === 0 ? `<td rowspan="${models.length}"><span class="task-name">${esc(t.label)}</span><span class="task-meta">${esc(t.size)}</span></td>` : ""}
        <td style="text-align:left"><span class="swatch" style="background:${color(i)}"></span> ${esc(m.label)}</td>
        ${t.type === "sts" ? cols.map(() => "<td>–</td>").join("") + cell("spearman") : cols.map(cell).join("") + "<td>–</td>"}</tr>`;
    });
  });
  models.forEach((m, i) => {
    h += `<tr>${i === 0 ? `<td rowspan="${models.length}"><span class="task-name">Main score</span><span class="task-meta">trung bình các task</span></td>` : ""}<td style="text-align:left"><span class="swatch" style="background:${color(i)}"></span> ${esc(m.label)}</td>
      <td colspan="${cols.length + 1}" class="${m.main_score === best && models.length > 1 ? "best" : ""}">${fmt(m.main_score, 2)}</td></tr>`;
  });
  document.getElementById("task-table").innerHTML = h + "</tbody>";
}

// Diverging delta chart: SAI - baseline (chọn baseline bằng nút)
let baseIdx = 1;
const dseg = document.getElementById("delta-seg");
dseg.innerHTML = models.length > 2 ? models.slice(1).map((m, j) =>
  `<button type="button" data-i="${j + 1}" aria-pressed="${j + 1 === baseIdx}">${esc(m.label)}</button>`).join("") : "";
dseg.addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  baseIdx = +b.dataset.i;
  dseg.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", +x.dataset.i === baseIdx));
  document.getElementById("delta-chart").innerHTML = ""; drawDelta();
});
function drawDelta() {
  const box = document.getElementById("delta-chart");
  const base = models[baseIdx];
  if (!base) { box.innerHTML = '<p class="empty">Cần ít nhất một baseline.</p>'; return; }
  document.getElementById("delta-note").textContent =
    `${ref.label} − ${base.label}, nDCG@10 / Spearman. Phải = SAI tốt hơn, trái = ${base.label} tốt hơn.`;
  const rows = tasks.map(t => ({ t, d: (score(ref, t, "ndcg@10") ?? NaN) - (score(base, t, "ndcg@10") ?? NaN) }))
    .filter(r => !isNaN(r.d));
  rows.push({ t: { label: "Main score" }, d: ref.main_score - base.main_score, main: true });
  const W = Math.max(280, box.clientWidth), labelW = 110, rowH = 30, H = rows.length * rowH + 24;
  const lim = Math.max(5, ...rows.map(r => Math.abs(r.d))) * 1.15;
  const mid = labelW + (W - labelW) / 2, half = (W - labelW) / 2 - 36;
  const x = v => mid + half * v / lim;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Chênh lệch theo task" }, box);
  el("line", { x1: mid, x2: mid, y1: 0, y2: H - 20, stroke: "var(--axis)" }, svg);
  el("text", { x: mid, y: H - 4, "text-anchor": "middle", class: "muted" }, svg).textContent = "0";
  rows.forEach((r, i) => {
    const y = i * rowH + 6, bh = 18;
    if (r.main) el("line", { x1: 0, x2: W, y1: y - 5, y2: y - 5, stroke: "var(--grid)" }, svg);
    const lt = el("text", { x: 0, y: y + bh / 2, "dominant-baseline": "middle" }, svg);
    lt.textContent = r.t.label; if (r.main) { lt.style.fontWeight = 700; lt.style.fill = "var(--text)"; }
    const good = r.d >= 0, w = Math.max(2, Math.abs(x(r.d) - mid));
    const d = good
      ? `M${mid},${y}h${w - 4}a4,4 0 0 1 4,4v${bh - 8}a4,4 0 0 1 -4,4h${-(w - 4)}z`
      : `M${mid},${y}h${-(w - 4)}a4,4 0 0 0 -4,4v${bh - 8}a4,4 0 0 0 4,4h${w - 4}z`;
    el("path", { d, fill: good ? color(0) : color(baseIdx) }, svg);
    const tx = el("text", { x: good ? x(r.d) + 6 : x(r.d) - 6, y: y + bh / 2, "dominant-baseline": "middle",
      "text-anchor": good ? "start" : "end", class: "val" }, svg);
    tx.textContent = (good ? "▲ +" : "▼ −") + fmt(Math.abs(r.d), 2);
  });
}

// Matryoshka line chart
function drawMrl() {
  const box = document.getElementById("mrl-chart");
  const dims = [...(ref.dims || [])].sort((a, b) => a - b);
  if (dims.length < 2) { box.innerHTML = '<p class="empty">Chỉ đo một số chiều — chạy với --dims 768 256 128 64 để xem.</p>'; return; }
  const mainAt = dim => {
    const v = tasks.map(t => score(ref, t, "ndcg@10", "dim" + dim)).filter(v => v != null);
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  };
  const pts = dims.map(d => ({ d, v: mainAt(d) })).filter(p => p.v != null);
  const all = pts.map(p => p.v).concat(models.slice(1).map(m => m.main_score));
  const lo = Math.max(0, Math.floor((Math.min(...all) - 5) / 5) * 5), hi = Math.min(100, Math.ceil((Math.max(...all) + 3) / 5) * 5);
  const W = Math.max(280, box.clientWidth), H = 260, L = 36, R = 16, T = 12, B = 30;
  const x = i => L + (W - L - R) * (pts.length === 1 ? 0.5 : i / (pts.length - 1));
  const y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Matryoshka" }, box);
  for (let t = lo; t <= hi; t += (hi - lo > 20 ? 10 : 5)) {
    el("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), stroke: "var(--grid)" }, svg);
    el("text", { x: L - 6, y: y(t), "text-anchor": "end", "dominant-baseline": "middle", class: "muted" }, svg).textContent = t;
  }
  let prevLabelY = -Infinity;
  models.slice(1).map((m, j) => ({ m, j })).sort((a, b) => b.m.main_score - a.m.main_score).forEach(({ m, j }) => {
    let ly = Math.max(y(m.main_score) - 6, prevLabelY + 14);  // tránh nhãn chồng nhau khi điểm gần nhau
    prevLabelY = ly;
    el("line", { x1: L, x2: W - R, y1: y(m.main_score), y2: y(m.main_score), stroke: color(j + 1), "stroke-width": 2, "stroke-dasharray": "6 4" }, svg);
    el("text", { x: W - R, y: ly, "text-anchor": "end", class: "muted" }, svg).textContent = `${m.label} (${m.dim}d) ${fmt(m.main_score)}`;
  });
  el("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.v)}`).join(""), fill: "none", stroke: color(0), "stroke-width": 2 }, svg);
  pts.forEach((p, i) => {
    el("text", { x: x(i), y: H - 8, "text-anchor": "middle", class: "muted" }, svg).textContent = p.d + "d";
    el("circle", { cx: x(i), cy: y(p.v), r: 5, fill: color(0), stroke: "var(--surface)", "stroke-width": 2 }, svg);
    if (i === 0 || i === pts.length - 1)
      el("text", { x: x(i), y: y(p.v) - 12, "text-anchor": i === 0 ? "start" : "end", class: "val" }, svg).textContent = fmt(p.v);
    const hit = el("rect", { x: x(i) - 24, y: T, width: 48, height: H - T - B, class: "hit" }, svg);
    hit.addEventListener("mousemove", e => showTip(e, `<b>${p.d} chiều</b>` + tipRow(0, ref.label, fmt(p.v, 2)) +
      `<div style="color:var(--muted)">Index 1M văn bản: ${fmt(p.d * 4 / 1024 / 1024 * 1e6 / 1024, 2)} GB (float32)</div>`));
    hit.addEventListener("mouseleave", hideTip);
  });
}

// Efficiency small multiples
function drawEff() {
  const dev = models[0].device;
  document.getElementById("speed-note").textContent =
    `Đo trên ${dev}, encode ${models[0].speed?.num_speed_docs ?? "–"} văn bản (tối đa 256 token). Tốc độ phụ thuộc phần cứng — chỉ so sánh tương đối.`;
  const charts = [
    { title: "Số tham số", unit: "M", get: m => m.params / 1e6, better: "thấp" },
    { title: "Số chiều vector", unit: "", get: m => m.dim, better: "thấp" },
    { title: "Tốc độ encode văn bản", unit: " vb/s", get: m => m.speed?.docs_per_s, better: "cao" },
    { title: "Độ trễ 1 query", unit: " ms", get: m => m.speed?.query_latency_ms, better: "thấp" },
  ];
  const box = document.getElementById("eff-charts"); box.innerHTML = "";
  const divs = charts.map(c => {
    const div = document.createElement("div");
    div.innerHTML = `<div style="font-weight:600;margin-bottom:2px">${c.title}</div><div class="note" style="margin:0 0 8px">${c.better === "cao" ? "Cao hơn" : "Thấp hơn"} là tốt hơn</div>`;
    box.appendChild(div);
    return div;
  });
  charts.forEach((c, ci) => {
    const div = divs[ci];  // đo bề rộng sau khi đã xếp đủ các ô vào grid
    const vals = models.map(c.get), max = Math.max(...vals.filter(v => v != null), 1e-9);
    const W = Math.max(200, div.clientWidth), barH = 18, gap = 6, H = models.length * (barH + gap);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": c.title }, div);
    models.forEach((m, i) => {
      const v = vals[i], y = i * (barH + gap), w = v == null ? 0 : Math.max(4, (W - 90) * v / max);
      if (v != null) el("path", { d: `M0,${y}h${w - 4}a4,4 0 0 1 4,4v${barH - 8}a4,4 0 0 1 -4,4h${-(w - 4)}z`, fill: color(i) }, svg);
      el("text", { x: w + 6, y: y + barH / 2, "dominant-baseline": "middle", class: "val" }, svg)
        .textContent = v == null ? "–" : (v >= 100 ? fmtInt(v) : fmt(v)) + c.unit;
      const hit = el("rect", { x: 0, y, width: W, height: barH, class: "hit" }, svg);
      hit.addEventListener("mousemove", e => showTip(e, `<b>${c.title}</b>` + tipRow(i, m.label, v == null ? "–" : fmt(v, 1) + c.unit)));
      hit.addEventListener("mouseleave", hideTip);
    });
  });
}

function drawAll() { drawTasks(); document.getElementById("delta-chart").innerHTML = ""; drawDelta();
  document.getElementById("mrl-chart").innerHTML = ""; drawMrl(); drawEff(); }
drawAll();
let lastW = window.innerWidth;
window.addEventListener("resize", () => { if (window.innerWidth !== lastW) { lastW = window.innerWidth; drawAll(); } });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
