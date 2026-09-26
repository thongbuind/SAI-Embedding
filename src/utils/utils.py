import torch

def select_device():
    """Trả về (device, amp_enabled, amp_dtype). AMP (bf16, hoặc fp16 trên GPU cũ) chỉ bật trên CUDA."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    return device, amp_enabled, amp_dtype

def get_step_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < total_steps * 0.4:
            return 1.0
        else:
            progress = (current_step - total_steps * 0.4) / (total_steps * 0.3)
            return max(0.1, 1.0 - 0.9 * progress)
    return lr_lambda

def log_progress(text):
    fixed_width = 82
    formatted_text = f"║ {text:<{fixed_width}} ║"
    print(formatted_text, flush=True)  # flush để log hiện ngay khi chạy nohup/ghi ra file

def memory_str(device) -> str:
    """VRAM đang dùng / đỉnh / reserved (GB); rỗng trên CPU."""
    if device.type == "cuda":
        return (f" | VRAM {torch.cuda.memory_allocated() / 1e9:.2f}GB (đỉnh {torch.cuda.max_memory_allocated() / 1e9:.2f}"
                f", reserved {torch.cuda.memory_reserved() / 1e9:.2f})")
    if device.type == "mps":
        return f" | VRAM {torch.mps.current_allocated_memory() / 1e9:.2f}GB (driver {torch.mps.driver_allocated_memory() / 1e9:.2f})"
    return ""

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m}m {s:.1f}s"
