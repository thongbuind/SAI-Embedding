import json
import random
import re
import hashlib
import unicodedata
from pathlib import Path
import numpy as np
import torch
import sentencepiece as spm
from src.utils.utils import log_progress

project_root = Path(__file__).resolve().parent.parent.parent
TOKENIZER_FILE = project_root / "src" / "tokenizer" / "tokenizer.model"

# SentencePiece của SAI không bật pad_id; id 0 là [UNK]. Model luôn nhận
# attention_mask tường minh nên dùng 0 để đệm vẫn an toàn.
PAD_ID = 0

_WHITESPACE = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    """NFC + lowercase + gộp khoảng trắng, khớp với dữ liệu tokenizer đã học.

    Tokenizer được train trên text NFC đã lowercase: cùng một câu ở dạng NFD
    (dấu tách rời) bị cắt thành số token gấp ~4 lần và gần như là OOV.
    """
    text = unicodedata.normalize("NFC", text)
    return _WHITESPACE.sub(" ", text).strip().lower()

def text_key(text: str) -> int:
    """Hash ổn định của text đã chuẩn hoá, dùng để phát hiện trùng lặp / false negative."""
    return int(hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()[:15], 16)

def resolve_path(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path

class EmbeddingTokenizer:
    def __init__(self, model_file: Path = TOKENIZER_FILE):
        self.model_file = str(model_file)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_file)
        self.bos_id = self.sp.piece_to_id("[BOS]")
        self.eos_id = self.sp.piece_to_id("[EOS]")

    # Cho phép pickle sang DataLoader worker (macOS dùng spawn).
    def __getstate__(self):
        return {"model_file": self.model_file}

    def __setstate__(self, state):
        self.__init__(state["model_file"])

    def encode(self, texts, prefix: str = "", max_len: int = 512):
        """[BOS] + tokens(prefix + text) + [EOS], cắt đuôi text nếu vượt max_len."""
        normalized = [normalize_text(prefix + t) for t in texts]
        pieces = self.sp.encode(normalized, out_type=int)
        return [[self.bos_id] + ids[:max_len - 2] + [self.eos_id] for ids in pieces]

    @staticmethod
    def pad(sequences, multiple_of: int = 8):
        """Right-padding. Trả về (input_ids, attention_mask, has_padding)."""
        max_len = max(len(s) for s in sequences)
        max_len = ((max_len + multiple_of - 1) // multiple_of) * multiple_of
        input_ids = torch.full((len(sequences), max_len), PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for i, s in enumerate(sequences):
            input_ids[i, :len(s)] = torch.as_tensor(s, dtype=torch.long)
            attention_mask[i, :len(s)] = 1
        has_padding = any(len(s) < max_len for s in sequences)
        return input_ids, attention_mask, has_padding

    def batch(self, texts, prefix: str = "", max_len: int = 512):
        return self.pad(self.encode(texts, prefix, max_len))

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def write_jsonl(path, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count

class PairDataset(torch.utils.data.Dataset):
    """Đọc nhiều file JSONL {query, positive, negatives?, type?} theo byte offset.

    Chỉ giữ offset trong RAM (không nạp toàn bộ text), mỗi worker tự mở file.
    `type == "symmetric"` (vd. NLI) nghĩa là hai vế cùng loại -> dùng query prefix cho cả hai.
    """
    def __init__(self, files: dict, max_rows_per_source: dict = None, seed: int = 54):
        max_rows_per_source = max_rows_per_source or {}
        rng = np.random.default_rng(seed)
        self.paths, self.sources = [], []
        offsets, file_idx = [], []

        for source, path in files.items():
            path = resolve_path(path)
            if not path.exists():
                log_progress(f"[data] Bỏ qua '{source}': không thấy {path}")
                continue
            source_offsets = []
            with open(path, "rb") as f:
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        source_offsets.append(pos)
            source_offsets = np.asarray(source_offsets, dtype=np.int64)
            cap = max_rows_per_source.get(source)
            if cap is not None and len(source_offsets) > cap:
                source_offsets = np.sort(rng.choice(source_offsets, size=cap, replace=False))
            self.paths.append(path)
            self.sources.append(source)
            offsets.append(source_offsets)
            file_idx.append(np.full(len(source_offsets), len(self.paths) - 1, dtype=np.int16))
            log_progress(f"[data] {source}: {len(source_offsets):,} mẫu")

        if not offsets:
            raise FileNotFoundError("Không có file train nào tồn tại")
        self.offsets = np.concatenate(offsets)
        self.source_idx = np.concatenate(file_idx)
        self._handles = {}

    def __len__(self):
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __getitem__(self, idx):
        file_id = int(self.source_idx[idx])
        handle = self._handles.get(file_id)
        if handle is None:
            handle = open(self.paths[file_id], "rb")
            self._handles[file_id] = handle
        handle.seek(int(self.offsets[idx]))
        row = json.loads(handle.readline().decode("utf-8"))
        row["source"] = self.sources[file_id]
        return row

    def source_counts(self):
        counts = np.bincount(self.source_idx, minlength=len(self.sources))
        return dict(zip(self.sources, counts.tolist()))

class SourceBatchSampler(torch.utils.data.Sampler):
    """Mỗi batch chỉ chứa mẫu của MỘT nguồn.

    In-batch negative cùng domain khó hơn (luật với luật, tin tức với tin tức) và
    không trộn task đối xứng (NLI) với bất đối xứng (query -> passage) trong một loss.
    Thứ tự batch giữa các nguồn được xáo, nên tỷ lệ nguồn tỷ lệ với số mẫu.
    `repeats` {chỉ số nguồn: r} cho nguồn nhỏ (vd. dữ liệu chuyên ngành) đi r lượt mỗi epoch; mỗi lượt
    xáo riêng nên một batch không bao giờ chứa hai bản của cùng một dòng.
    """
    def __init__(self, source_idx, batch_size: int, seed: int = 54, repeats: dict = None):
        self.source_idx = np.asarray(source_idx)
        self.batch_size = batch_size
        self.seed = seed
        self.repeats = repeats or {}
        self.epoch = 0
        self.start_batch = 0

    def set_epoch(self, epoch: int, start_batch: int = 0):
        self.epoch = epoch
        self.start_batch = start_batch

    def _batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for source in np.unique(self.source_idx):
            members = np.flatnonzero(self.source_idx == source)
            for _ in range(self.repeats.get(int(source), 1)):
                idx = rng.permutation(members)
                n_full = len(idx) // self.batch_size
                if n_full == 0:
                    # Nguồn nhỏ hơn 1 batch vẫn được giữ lại thành một batch nhỏ.
                    batches.append(idx.tolist())
                    continue
                for b in range(n_full):
                    batches.append(idx[b * self.batch_size:(b + 1) * self.batch_size].tolist())
        order = rng.permutation(len(batches))
        return [batches[i] for i in order]

    def __len__(self):
        return len(self._batches())

    def batches_per_source(self, names):
        counts = np.bincount([self.source_idx[b[0]] for b in self._batches()], minlength=len(names))
        return dict(zip(names, counts.tolist()))

    def __iter__(self):
        yield from self._batches()[self.start_batch:]

class PairCollator:
    """`max_passage_len_by_source` ghi đè độ dài passage cho từng nguồn (vd. chunk tài liệu dài ~335
    token, cắt ở 256 thì mất đoạn chứa đáp án)."""
    def __init__(self, tokenizer: EmbeddingTokenizer, query_prefix: str, passage_prefix: str,
                 max_query_len: int, max_passage_len: int, num_negatives: int,
                 max_passage_len_by_source: dict = None):
        self.tokenizer = tokenizer
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len
        self.max_passage_len_by_source = max_passage_len_by_source or {}
        self.num_negatives = num_negatives

    def __call__(self, rows):
        symmetric = rows[0].get("type") == "symmetric"
        passage_prefix = self.query_prefix if symmetric else self.passage_prefix
        max_passage_len = self.max_passage_len_by_source.get(rows[0]["source"], self.max_passage_len)

        queries = [r["query"] for r in rows]
        positives = [r["positive"] for r in rows]
        negatives = []
        for r in rows:
            negs = [n for n in (r.get("negatives") or []) if n]
            if len(negs) > self.num_negatives:
                negs = random.sample(negs, self.num_negatives)
            negatives.extend(negs)
        # passages = [positive của từng query] + [toàn bộ hard negative của batch]
        passages = positives + negatives

        q_ids, q_mask, q_pad = self.tokenizer.batch(queries, self.query_prefix, self.max_query_len)
        p_ids, p_mask, p_pad = self.tokenizer.batch(passages, passage_prefix, max_passage_len)
        return {
            "q_ids": q_ids, "q_mask": q_mask, "q_has_padding": q_pad,
            "p_ids": p_ids, "p_mask": p_mask, "p_has_padding": p_pad,
            "q_keys": torch.tensor([text_key(t) for t in queries], dtype=torch.long),
            "p_keys": torch.tensor([text_key(t) for t in passages], dtype=torch.long),
            # Nhóm của positive (trường "group", vd. mục tài liệu); negative không rõ nhóm -> -1.
            "p_groups": torch.tensor([text_key(f"group:{r['group']}") if r.get("group") else -1 for r in rows]
                                     + [-1] * len(negatives), dtype=torch.long),
            "source": rows[0]["source"],
        }
