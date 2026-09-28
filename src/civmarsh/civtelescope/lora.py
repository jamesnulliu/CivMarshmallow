"""LoRA adapters for CivTelescope and merging them into the base weights.

Each adapted linear layer computes ``base(x) + (x @ A.T @ B.T) * alpha / r``
with A (r x in) Kaiming-uniform initialised and B (out x r) zero-initialised,
so training starts from the base model. The base model stays in bf16 and is
frozen; A and B are fp32 on the base layer's device. No dropout.

Adapters are indexed by the order in which ``model.named_modules()`` visits the
target projections; a checkpoint is ``{"<i>": {"A": tensor, "B": tensor}}`` in
that order, and merging walks the same order.

torch is imported on first use, so this module imports without it.
"""

from __future__ import annotations

from contextlib import contextmanager

DEFAULT_RANK = 16
DEFAULT_ALPHA = 32
DEFAULT_TARGETS = ("q_proj", "v_proj")

_LORA_LINEAR = None


def lora_linear_class():
    """The ``LoRALinear`` module class (built on first use)."""
    global _LORA_LINEAR
    if _LORA_LINEAR is not None:
        return _LORA_LINEAR
    import math

    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, r: int = DEFAULT_RANK, alpha=DEFAULT_ALPHA):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.scale = alpha / r
            self.A = nn.Parameter(torch.zeros(r, base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.enabled = True

        def forward(self, x):
            out = self.base(x)
            if self.enabled:
                lora = (x.to(self.A.dtype) @ self.A.T @ self.B.T) * self.scale
                out = out + lora.to(out.dtype)
            return out

    _LORA_LINEAR = LoRALinear
    return LoRALinear


def inject_lora(
    model, r: int = DEFAULT_RANK, alpha=DEFAULT_ALPHA, targets=DEFAULT_TARGETS
) -> list:
    """Wrap every target projection of `model` in place; returns the adapters
    in checkpoint order."""
    import torch
    from torch import nn

    cls = lora_linear_class()
    wrapped = []
    for _name, mod in model.named_modules():
        for t in targets:
            child = getattr(mod, t, None)
            if isinstance(child, nn.Linear):
                lo = cls(child, r=r, alpha=alpha)
                dev = child.weight.device
                lo.A.data = lo.A.data.to(device=dev, dtype=torch.float32)
                lo.B.data = lo.B.data.to(device=dev, dtype=torch.float32)
                setattr(mod, t, lo)
                wrapped.append(lo)
    return wrapped


@contextmanager
def adapters_disabled(loras):
    """Run the bare backbone: the adapters contribute nothing inside."""
    for m in loras:
        m.enabled = False
    try:
        yield
    finally:
        for m in loras:
            m.enabled = True


def adapter_state(loras) -> dict:
    """CPU copy of the adapter weights in checkpoint layout."""
    return {
        f"{i}": {"A": m.A.detach().cpu(), "B": m.B.detach().cpu()}
        for i, m in enumerate(loras)
    }


def load_adapter_state(loras, state: dict) -> None:
    """Copy checkpoint weights into injected adapters (same traversal order)."""
    import torch

    if len(state) != len(loras):
        raise ValueError(f"adapter count {len(loras)} != checkpoint {len(state)}")
    for i, m in enumerate(loras):
        w = state[str(i)]
        m.A.data.copy_(w["A"].to(m.A.device, torch.float32))
        m.B.data.copy_(w["B"].to(m.B.device, torch.float32))


def merge_adapters(model, state: dict, *, alpha=DEFAULT_ALPHA, targets=DEFAULT_TARGETS):
    """Add ``B @ A * alpha / r`` into each target weight of an unwrapped
    `model`, in the traversal order the adapters were injected in. The rank is
    read from the checkpoint. Returns the number of merged layers."""
    from torch import nn

    idx = 0
    for name, mod in model.named_modules():
        for t in targets:
            child = getattr(mod, t, None)
            if not isinstance(child, nn.Linear):
                continue
            A = state[str(idx)]["A"].float()  # [r, in]
            B = state[str(idx)]["B"].float()  # [out, r]
            r = A.shape[0]
            if A.shape != (r, child.in_features) or B.shape != (child.out_features, r):
                raise ValueError(
                    f"{name}.{t}: shapes {tuple(A.shape)} {tuple(B.shape)}"
                )
            delta = (B @ A) * (alpha / r)
            child.weight.data += delta.to(child.weight.dtype)
            idx += 1
    if str(idx) in state:
        raise ValueError(f"checkpoint has more adapters than model targets ({idx})")
    return idx


def merge_checkpoint(base_model: str, adapters_path, out_dir, *, alpha=DEFAULT_ALPHA):
    """Merge an adapter checkpoint into `base_model` and save a Hugging Face
    directory (weights + tokenizer) that an inference server can load."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    state = torch.load(adapters_path, map_location="cpu")
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(base_model)
    n = merge_adapters(model, state, alpha=alpha)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return n
