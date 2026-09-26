"""Tải sổ tay tiếng Việt của mọi mẫu xe từ cổng chính hãng om.vinfastauto.com (API công khai).

Mỗi mẫu xe lấy đời mới nhất. Mẫu có sổ tay dạng web được lưu thành một file JSON gồm cây mục
lục và HTML từng mục (API chỉ trả HTML của một chương mỗi lần gọi, qua tham số `chapter`).
Mẫu đời cũ chỉ có PDF thì tải PDF.

    python -m src.data.fetch_vinfast_manuals
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://omapi.vinfastauto.com/fe/v1"
OUT_DIR = Path("data/embedding/manual_web")
HEADERS = {"User-Agent": "Mozilla/5.0"}


def get(url: str, retries: int = 3) -> bytes:
    for i in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60).read()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(3 * (i + 1))


def api(path: str, **params) -> dict:
    return json.loads(get(f"{API}{path}?{urllib.parse.urlencode(params)}"))


def fetch_model(name: str, version: str, sleep: float):
    base = {"content": "chapter", "carModel": name, "version": version, "lang": "vi", "country": "vn"}
    menu = api("/menu", **base)
    stem = f"{name.replace(' ', '_')}_{version}"
    if not menu.get("data"):
        pdf = menu.get("pdf")
        if not pdf:
            print(f"[bỏ] {name} {version}: không có nội dung")
            return
        path = OUT_DIR / f"{stem}.pdf"
        if not path.exists():
            path.write_bytes(get(pdf))
        print(f"[pdf] {name} {version}: {path.stat().st_size / 1e6:.1f}MB")
        return

    path = OUT_DIR / f"{stem}.json"
    if path.exists():
        print(f"[có sẵn] {path}")
        return
    chapters = []
    for lv1 in menu["data"]:
        for lv2 in lv1.get("childs") or []:
            html = lv2.get("html")
            if not html:
                data = api("/menu", **base, chapter=lv2["id"])["data"]
                hit = next((c for n in data for c in n.get("childs") or [] if c["id"] == lv2["id"]), None)
                html = (hit or {}).get("html")
                time.sleep(sleep)
            chapters.append({"lv1": lv1["name"], "lv2": lv2["name"], "id": lv2["id"],
                             "level3": [x["name"] for x in lv2.get("level3") or []], "html": html or ""})
    path.write_text(json.dumps({"model": name, "version": version, "chapters": chapters}, ensure_ascii=False))
    print(f"[web] {name} {version}: {len(chapters)} mục, {path.stat().st_size / 1e6:.1f}MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="model_name trong /carModel; mặc định tất cả")
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = api("/carModel", lang="vi", country="vn")["data"]["models"]
    for group in models.values():
        for m in group:
            if args.models and m["model_name"] not in args.models:
                continue
            try:
                fetch_model(m["model_name"], m["versions"][0], args.sleep)
            except Exception as e:
                print(f"[lỗi] {m['model_name']}: {e}")


if __name__ == "__main__":
    main()
