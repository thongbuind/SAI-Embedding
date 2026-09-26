import torch
import torch.nn.functional as F

def false_negative_masks(q_keys: torch.Tensor, p_keys: torch.Tensor, p_groups: torch.Tensor = None):
    """Mask các cặp là "negative giả" trong batch.

    q_keys: (B,) hash query; p_keys: (P,) hash passage, P >= B, p_keys[:B] là positive.
    p_groups: (P,) hash nhóm của passage (vd. cùng một mục tài liệu), -1 = không rõ. Hai passage cùng
      nhóm coi như trùng: cặp tự sinh từ một đoạn có positive chữ hơi khác nhau (bỏ câu/tiêu đề).
    - q->p: passage j bị mask cho query i nếu j != i và (text/nhóm j trùng positive i,
      hoặc j là positive của một dòng khác có cùng query với i).
    - p->q: query j bị mask cho positive i nếu j != i và (query trùng, hoặc positive trùng text/nhóm).
    """
    B, P = q_keys.size(0), p_keys.size(0)
    device = q_keys.device
    eye_bp = torch.zeros(B, P, dtype=torch.bool, device=device)
    eye_bp[:, :B] = torch.eye(B, dtype=torch.bool, device=device)

    same_passage = p_keys[:B, None] == p_keys[None, :]                  # (B, P)
    if p_groups is not None:
        same_passage |= (p_groups[:B, None] == p_groups[None, :]) & (p_groups[:B, None] >= 0)
    same_query = torch.zeros(B, P, dtype=torch.bool, device=device)
    same_query[:, :B] = q_keys[:, None] == q_keys[None, :]
    q2p_mask = (same_passage | same_query) & ~eye_bp

    eye_bb = torch.eye(B, dtype=torch.bool, device=device)
    p2q_mask = ((q_keys[:, None] == q_keys[None, :]) | same_passage[:, :B]) & ~eye_bb
    return q2p_mask, p2q_mask

def info_nce(q: torch.Tensor, p: torch.Tensor, q2p_mask, p2q_mask,
             temperature: float, symmetric: bool = True) -> torch.Tensor:
    """q: (B, d), p: (P, d) đã L2-normalize; p[:B] là positive của từng query."""
    B = q.size(0)
    targets = torch.arange(B, device=q.device)
    logits = (q @ p.T) / temperature
    logits = logits.masked_fill(q2p_mask, float("-inf"))
    loss = F.cross_entropy(logits, targets)
    if symmetric:
        logits_p2q = (p[:B] @ q.T) / temperature
        logits_p2q = logits_p2q.masked_fill(p2q_mask, float("-inf"))
        loss = 0.5 * (loss + F.cross_entropy(logits_p2q, targets))
    return loss

def matryoshka_info_nce(q_raw: torch.Tensor, p_raw: torch.Tensor, q_keys, p_keys,
                        dims, temperature: float, symmetric: bool = True, p_groups=None,
                        weights=None) -> torch.Tensor:
    """InfoNCE trung bình (có trọng số) trên các prefix chiều (Matryoshka). q_raw/p_raw CHƯA normalize.
    weights: trọng số từng chiều theo thứ tự dims, None = bằng nhau."""
    q2p_mask, p2q_mask = false_negative_masks(q_keys, p_keys, p_groups)
    q_raw, p_raw = q_raw.float(), p_raw.float()
    losses = []
    for dim in dims:
        q = F.normalize(q_raw[:, :dim], dim=-1)
        p = F.normalize(p_raw[:, :dim], dim=-1)
        losses.append(info_nce(q, p, q2p_mask, p2q_mask, temperature, symmetric))
    losses = torch.stack(losses)
    if weights is None:
        return losses.mean()
    w = torch.tensor(weights, dtype=losses.dtype, device=losses.device)
    return (losses * w).sum() / w.sum()

class RandContext:
    """Lưu/khôi phục trạng thái RNG để lần forward thứ hai của GradCache
    sinh đúng dropout mask như lần forward không gradient."""
    def __init__(self, device: torch.device):
        self.device = device
        self.cpu_state = torch.get_rng_state()
        if device.type == "cuda":
            self.device_state = torch.cuda.get_rng_state(device)
        elif device.type == "mps":
            self.device_state = torch.mps.get_rng_state()
        else:
            self.device_state = None

    def __enter__(self):
        self._saved_cpu = torch.get_rng_state()
        torch.set_rng_state(self.cpu_state)
        if self.device.type == "cuda":
            self._saved_device = torch.cuda.get_rng_state(self.device)
            torch.cuda.set_rng_state(self.device_state, self.device)
        elif self.device.type == "mps":
            self._saved_device = torch.mps.get_rng_state()
            torch.mps.set_rng_state(self.device_state)

    def __exit__(self, *exc):
        torch.set_rng_state(self._saved_cpu)
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(self._saved_device, self.device)
        elif self.device.type == "mps":
            torch.mps.set_rng_state(self._saved_device)

def _chunks(ids, mask, chunk_size, device):
    """Chia batch (tensor CPU) thành chunk, cắt cột padding thừa rồi mới chuyển sang device."""
    for start in range(0, ids.size(0), chunk_size):
        chunk_mask = mask[start:start + chunk_size]
        width = int(chunk_mask.sum(dim=1).max())
        width = min(ids.size(1), ((width + 7) // 8) * 8)
        chunk_mask = chunk_mask[:, :width]
        has_padding = bool((chunk_mask == 0).any())
        yield (ids[start:start + chunk_size, :width].to(device, non_blocking=True),
               chunk_mask.to(device, non_blocking=True), has_padding)

def grad_cache_step(model, batch, device, amp_ctx, loss_fn, chunk_size: int, loss_scale: float = 1.0,
                    token_budget: int = None):
    """GradCache (Gao et al., 2021): batch contrastive lớn với VRAM của một chunk.

    1) Forward không gradient từng chunk -> embedding của cả batch.
    2) Tính loss trên embedding (có grad) -> gradient theo embedding.
    3) Forward lại từng chunk có gradient, backward với gradient đã cache.
    Kết quả gradient tương đương forward cả batch một lần.

    token_budget: số token tối đa mỗi chunk (số dòng x độ dài đã pad). Batch có passage dài hơn
    mức thường (vd. nguồn có max_passage_len 512) tự chia chunk ít dòng hơn, VRAM giữ như cũ.
    """
    def embed_no_grad(ids, mask):
        rows = chunk_size if not token_budget else max(1, min(chunk_size, token_budget // ids.size(1)))
        reps, states, chunks = [], [], []
        for chunk in _chunks(ids, mask, rows, device):
            states.append(RandContext(device))
            with torch.no_grad(), amp_ctx():
                reps.append(model(*chunk, normalize=False).float())
            chunks.append(chunk)
        return torch.cat(reps), states, chunks, rows

    q_reps, q_states, q_chunks, q_rows = embed_no_grad(batch["q_ids"], batch["q_mask"])
    p_reps, p_states, p_chunks, p_rows = embed_no_grad(batch["p_ids"], batch["p_mask"])

    q_reps = q_reps.detach().requires_grad_()
    p_reps = p_reps.detach().requires_grad_()
    loss = loss_fn(q_reps, p_reps)
    (loss * loss_scale).backward()

    for reps, states, chunks, rows in ((q_reps, q_states, q_chunks, q_rows), (p_reps, p_states, p_chunks, p_rows)):
        grads = reps.grad.split(rows)
        for state, chunk, grad in zip(states, chunks, grads):
            with state, amp_ctx():
                rep = model(*chunk, normalize=False).float()
            torch.dot(rep.flatten(), grad.flatten()).backward()
    return loss.detach()

def direct_step(model, batch, device, amp_ctx, loss_fn, loss_scale: float = 1.0):
    """Forward/backward thẳng cả batch (khi batch vừa VRAM, không cần GradCache)."""
    with amp_ctx():
        q = model(batch["q_ids"].to(device), batch["q_mask"].to(device),
                  batch["q_has_padding"], normalize=False)
        p = model(batch["p_ids"].to(device), batch["p_mask"].to(device),
                  batch["p_has_padding"], normalize=False)
    loss = loss_fn(q, p)
    (loss * loss_scale).backward()
    return loss.detach()
