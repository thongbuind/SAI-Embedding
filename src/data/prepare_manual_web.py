"""Tách sổ tay web (tải bởi fetch_vinfast_manuals) thành chunk và văn bản MNTP.

Đầu ra:
- data/embedding/manual_web/chunks.jsonl: chunk theo mục cấp 3 (Detail-Heading), mục dài được chia
  theo Sub-Section để mỗi chunk không quá CHUNK_WORDS từ. Passage dạng `chương › mục › tiểu mục`
  + nội dung, giống manual_vf8. Chunk trùng hệt nhau giữa các xe (xe Green dùng lại sổ tay VF 5...)
  mang cùng `text_key`, để chỉ phải sinh câu hỏi một lần.
- data/embedding/manual_web/clusters.jsonl: gom chunk gần trùng giữa các xe (cosine trigram từ
  ≥ CLUSTER_SIM), đại diện lấy theo thứ tự PRIORITY. Câu hỏi chỉ viết cho đại diện rồi dùng chung
  cho cả cụm (build_manual_web_pairs vẫn lọc lại bằng BM25 theo từng xe).
- data/embedding/mntp_sources/manual_web.jsonl: mỗi mục cấp 2 một văn bản, bỏ văn bản trùng giữa
  các xe. Bỏ VF 8 và VF e34 vì đã có trong mntp_sources/manual.jsonl. Thêm chữ trích từ các sổ
  tay PDF đời cũ trong MNTP_PDFS (chỉ MNTP, không sinh câu hỏi). Fadil, Lux A2.0/SA2.0 không có lớp
  chữ (chữ là ảnh/nét vẽ, cần OCR) nên không dùng.

    python -m src.data.prepare_manual_web
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from pypdf import PdfReader
from scipy.sparse import csr_matrix

WEB_DIR = Path("data/embedding/manual_web")
MNTP_OUT = Path("data/embedding/mntp_sources/manual_web.jsonl")
CHUNK_WORDS = 220
MNTP_SKIP = {"VF8", "VF e34"}
QUESTION_SKIP = {"VF8", "VF e34"}
PRIORITY = ["VF3", "VF5", "VF6", "VF7", "VF9", "VF MPV 7", "LacHong900LX", "VF2", "VF8NP",
            "Limo Green", "Herio Green", "VF5 BCO3", "Nerio Green", "Minio Green", "EC VAN", "EB 6", "EB 8"]
CLUSTER_SIM = 0.8
MNTP_PDFS = {"President_2020.pdf": "President", "EB_10_2021.pdf": "EB 10"}
PDF_DOC_WORDS = 300
SEP = " › "

HEADING_CLASSES = {"Detail-Heading"}
SUB_CLASSES = {"Sub-Section"}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def block_text(el) -> str:
    """Một phần tử cấp cao của HTML sổ tay → văn bản (bảng thành các dòng `a | b`)."""
    if el.name == "table":
        rows = []
        for tr in el.find_all("tr"):
            cells = [norm(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)
    if el.name in ("ul", "ol"):
        items = [norm(li.get_text(" ")) for li in el.find_all("li", recursive=False)]
        if el.name == "ol":
            start = int(el.get("start") or 1) if str(el.get("start") or "1").isdigit() else 1
            return "\n".join(f"{i}. {t}" for i, t in enumerate(items, start) if t)
        return "\n".join(f"- {t}" for t in items if t)
    if el.name == "section":
        return "\n\n".join(t for t in (block_text(c) for c in el.find_all(recursive=False)) if t)
    for br in el.find_all("br"):
        br.replace_with("\n")
    text = "\n".join(norm(line) for line in el.get_text().split("\n"))
    return re.sub(r"\n{2,}", "\n", text).strip()


def parse_chapter(html: str):
    """HTML một mục cấp 2 → [(tiêu đề cấp 3, [(tiêu đề Sub-Section hoặc None, [đoạn...])])]."""
    soup = BeautifulSoup(html, "html.parser")
    root = soup.body or soup
    sections = []
    for el in root.find_all(recursive=False):
        classes = set(el.get("class") or [])
        if "Image" in classes or el.name == "img":
            continue
        text = block_text(el)
        if not text:
            continue
        if classes & HEADING_CLASSES:
            sections.append((text, []))
            continue
        if not sections:
            sections.append(("", []))
        subs = sections[-1][1]
        if classes & SUB_CLASSES:
            subs.append((text, []))
        else:
            if not subs:
                subs.append((None, []))
            subs[-1][1].append(text)
    return sections


def split_long(title, paras):
    """Sub-Section dài hơn CHUNK_WORDS → nhiều phần, cắt theo đoạn (đoạn quá dài cắt theo dòng)."""
    lines = [l for p in paras for l in p.split("\n")]
    parts, cur, n = [], [], 0
    for line in lines:
        w = len(line.split())
        if cur and n + w > CHUNK_WORDS:
            parts.append(cur)
            cur, n = [], 0
        cur.append(line)
        n += w
    if cur:
        parts.append(cur)
    return [(title, "\n".join(([title] if title else []) + p)) for p in parts]


def chunk_section(subs):
    """Gộp các Sub-Section liên tiếp thành chunk ≤ CHUNK_WORDS từ (Sub-Section dài thì bị cắt)."""
    pieces = [piece for title, paras in subs for piece in split_long(title, paras)]
    chunks, cur, cur_words = [], [], 0
    for title, body in pieces:
        words = len(body.split())
        if cur and cur_words + words > CHUNK_WORDS:
            chunks.append(cur)
            cur, cur_words = [], 0
        cur.append((title, body))
        cur_words += words
    if cur:
        chunks.append(cur)
    return chunks


def key(text: str) -> str:
    return hashlib.md5(norm(text).lower().encode()).hexdigest()[:16]


def write_clusters(chunks):
    rank = {m: i for i, m in enumerate(PRIORITY)}
    uniq = {}
    for c in sorted((c for c in chunks if c["model"] not in QUESTION_SKIP),
                    key=lambda c: rank.get(c["model"], len(rank))):
        uniq.setdefault(c["text_key"], c)
    keys = list(uniq)
    vocab, rows, cols = {}, [], []
    for i, k in enumerate(keys):
        words = re.findall(r"\w+", uniq[k]["text"].split("\n", 1)[1].lower())
        grams = {" ".join(words[j:j + 3]) for j in range(len(words) - 2)} or {" ".join(words)}
        for g in grams:
            rows.append(i)
            cols.append(vocab.setdefault(g, len(vocab)))
    x = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(keys), len(vocab)))
    x = csr_matrix(x.multiply(1 / np.sqrt(np.asarray(x.sum(axis=1)))))
    sim = (x @ x.T).tocsr()
    rep = {}
    for i in range(len(keys)):
        lo, hi = sim.indptr[i], sim.indptr[i + 1]
        near = [j for j, v in zip(sim.indices[lo:hi], sim.data[lo:hi]) if v >= CLUSTER_SIM and j < i and rep[j] == j]
        rep[i] = min(near) if near else i
    members = {}
    for i, r in rep.items():
        members.setdefault(r, []).append(i)
    by_key = {}
    for c in chunks:
        by_key.setdefault(c["text_key"], set()).add(c["model"])
    with open(WEB_DIR / "clusters.jsonl", "w") as f:
        for r, ms in members.items():
            c = uniq[keys[r]]
            f.write(json.dumps({"key": keys[r], "members": [keys[i] for i in ms],
                                "models": sorted({m for i in ms for m in by_key[keys[i]]}),
                                "rep_model": c["model"], "text": c["text"]}, ensure_ascii=False) + "\n")
    return len(members)


def clean_pdf_page(text: str) -> str:
    """Chữ một trang PDF: bỏ số trang, dòng rác, sửa tiêu đề in hoa lỗi font ("cỬA SỔ"), nối dòng ngắt giữa câu."""
    text = re.sub(r"^\s*\d{1,3}(?=\D)", "", text.replace("\xa0", " "))
    lines = []
    for raw in text.split("\n"):
        line = re.sub(r"[■►•]\s*", "- ", norm(raw)).strip()
        letters = [ch for ch in line if ch.isalpha()]
        if len(letters) < 3 or re.fullmatch(r"[\d\s./-]+", line):
            continue
        if sum(ch.isupper() for ch in letters) >= 0.7 * len(letters):
            line = line.upper()
        if lines and not re.search(r"[.:;!?]$", lines[-1]) and line[0].islower():
            lines[-1] += " " + line
        else:
            lines.append(line)
    return "\n".join(lines)


def pdf_docs(path: Path, model: str):
    """Gộp các trang liên tiếp đến khoảng PDF_DOC_WORDS từ thành một văn bản MNTP."""
    docs, buf = [], []
    for page in PdfReader(path).pages:
        text = clean_pdf_page(page.extract_text() or "")
        if len(text.split()) < 20:
            continue
        buf.append(text)
        if sum(len(t.split()) for t in buf) >= PDF_DOC_WORDS:
            docs.append("\n".join(buf))
            buf = []
    if buf:
        docs.append("\n".join(buf))
    return [{"text": d, "model": model} for d in docs]


def main():
    chunks, mntp, seen_mntp = [], [], set()
    for path in sorted(WEB_DIR.glob("*.json")):
        manual = json.loads(path.read_text())
        if "chapters" not in manual:  # pairs_stats.json...
            continue
        model = manual["model"]
        n_before = len(chunks)
        for ch in manual["chapters"]:
            if not ch["html"]:
                continue
            sections = parse_chapter(ch["html"])
            doc_parts = []
            for lv3, subs in sections:
                lv3 = lv3 or ch["lv2"]
                path3 = SEP.join([ch["lv1"], ch["lv2"], lv3])
                doc_parts.append(lv3 + "\n" + "\n".join("\n".join(([t] if t else []) + p) for t, p in subs))
                for i, part in enumerate(chunk_section(subs)):
                    body = "\n".join(b for _, b in part).strip()
                    if len(body.split()) < 8:
                        continue
                    subtitles = [t for t, _ in part if t]
                    chunks.append({
                        "id": f"{model}#{ch['id']}#{len(chunks) - n_before:04d}",
                        "model": model, "version": manual["version"],
                        "group": SEP.join([ch["lv2"], lv3]), "subsections": subtitles,
                        "text": f"{path3}\n{body}", "text_key": key(body),
                    })
            doc = f"{ch['lv1']}{SEP}{ch['lv2']}\n" + "\n\n".join(doc_parts)
            k = key(doc)
            if model not in MNTP_SKIP and k not in seen_mntp and len(doc.split()) >= 30:
                seen_mntp.add(k)
                mntp.append({"text": doc, "model": model})
        print(f"{model:14s} {len(chunks) - n_before:5d} chunk")
    for name, model in MNTP_PDFS.items():
        if (WEB_DIR / name).exists():
            docs = pdf_docs(WEB_DIR / name, model)
            mntp += docs
            print(f"{model:14s} PDF: {len(docs)} văn bản MNTP, {sum(len(d['text'].split()) for d in docs):,} từ")

    with open(WEB_DIR / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(MNTP_OUT, "w") as f:
        for d in mntp:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    n_clusters = write_clusters(chunks)
    uniq = len({c["text_key"] for c in chunks})
    words = sum(len(d["text"].split()) for d in mntp)
    print(f"chunks: {len(chunks)} ({uniq} nội dung khác nhau, {n_clusters} cụm cần câu hỏi) | MNTP: {len(mntp)} văn bản, {words:,} từ")


if __name__ == "__main__":
    main()
