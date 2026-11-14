"""Frozen DistilBERT classifier head and a printable layer summary."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.config import AttrDict, load_config
from src.embed import load_encoder
from src.utils import ensure_dirs, get_logger, set_seed

CLS_DIM = 768


class ClassifierHead(nn.Module):
    """Linear(768, hidden_units) -> ReLU -> Dropout -> Linear(hidden_units, num_classes)."""

    def __init__(self, config: AttrDict) -> None:
        super().__init__()
        hidden_units = int(config.model.hidden_units)
        dropout = float(config.model.dropout)
        num_classes = int(config.model.num_classes)
        self.fc1 = nn.Linear(CLS_DIM, hidden_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_units, num_classes)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of 768-d [CLS] vectors."""
        hidden = self.relu(self.fc1(cls_embedding))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


class DistilBertClassifier(nn.Module):
    """Frozen DistilBERT encoder plus ClassifierHead, used at inference time."""

    def __init__(self, config: AttrDict | None = None, encoder: nn.Module | None = None) -> None:
        super().__init__()
        cfg = config if config is not None else load_config()
        self.config = cfg
        if encoder is None:
            encoder, device = load_encoder(cfg)
        else:
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
            device = str(next(encoder.parameters()).device)
        self.encoder = encoder
        self.head = ClassifierHead(cfg)
        self.to(device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return class logits from the frozen encoder's [CLS] vector."""
        with torch.no_grad():
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            cls_vec = hidden[:, 0, :]
        return self.head(cls_vec)


def count_parameters(module: nn.Module, trainable_only: bool) -> int:
    """Return the number of parameters in module, optionally only those with requires_grad."""
    total = 0
    for param in module.parameters():
        if (not trainable_only) or param.requires_grad:
            total += int(param.numel())
    return total


def _is_leaf(module: nn.Module) -> bool:
    """Return True if module has no child modules."""
    return not list(module.children())


def _output_shape(output: object) -> tuple[int, ...] | None:
    """Best-effort tensor shape from a forward hook output."""
    tensor = output
    if isinstance(output, (tuple, list)) and output:
        tensor = output[0]
    if hasattr(tensor, "last_hidden_state"):
        tensor = tensor.last_hidden_state
    if hasattr(tensor, "shape"):
        return tuple(int(dim) for dim in tensor.shape)
    return None


def _layer_trainable(module: nn.Module) -> bool:
    """Return True if any parameter of this module (not its children) requires grad."""
    return any(param.requires_grad for param in module.parameters(recurse=False))


def _dummy_forward(model: nn.Module, config: AttrDict) -> torch.Tensor:
    """Run a tiny forward pass so hooks can record output shapes."""
    device = next(model.parameters()).device
    if isinstance(model, DistilBertClassifier):
        length = int(config.model.max_length)
        input_ids = torch.zeros(2, length, dtype=torch.long, device=device)
        input_ids[:, 0] = 101
        attention_mask = torch.zeros_like(input_ids)
        attention_mask[:, 0] = 1
        return model(input_ids, attention_mask)
    embeddings = torch.zeros(2, CLS_DIM, device=device)
    return model(embeddings)


def _summary_table(model: nn.Module, config: AttrDict) -> str:
    """Build a layer table: name, output shape, parameter count, trainable flag."""
    shapes: dict[str, tuple[int, ...] | None] = {}
    hooks = []
    for name, module in model.named_modules():
        if not name or not _is_leaf(module):
            continue
        hooks.append(
            module.register_forward_hook(
                lambda _mod, _inp, output, layer=name: shapes.__setitem__(layer, _output_shape(output))
            )
        )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _dummy_forward(model, config)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    rows: list[tuple[str, str, str, str]] = []
    for name, module in model.named_modules():
        if not name or not _is_leaf(module):
            continue
        n_params = count_parameters(module, trainable_only=False)
        rows.append(
            (
                name,
                str(_format_shape(shapes.get(name))),
                f"{n_params:,}",
                "yes" if _layer_trainable(module) else "no",
            )
        )

    name_w = max(56, max((len(row[0]) for row in rows), default=5))
    shape_w = max(18, max((len(row[1]) for row in rows), default=12))
    param_w = max(12, max((len(row[2]) for row in rows), default=6))
    train_w = 10
    header = f"{'Layer':<{name_w}} {'Output shape':<{shape_w}} {'Params':>{param_w}} {'Trainable':>{train_w}}"
    rule = "-" * len(header)
    lines = [header, rule]
    for name, shape, params, trainable in rows:
        lines.append(f"{name:<{name_w}} {shape:<{shape_w}} {params:>{param_w}} {trainable:>{train_w}}")
    lines.append(rule)

    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    frozen = total - trainable
    lines.append(f"Total params          {total:,}")
    lines.append(f"Trainable params      {trainable:,}")
    lines.append(f"Non-trainable params  {frozen:,}")
    return "\n".join(lines) + "\n"


def _format_shape(shape: tuple[int, ...] | None) -> str:
    """Render a shape tuple as '(d0, d1, ...)'."""
    if shape is None:
        return "-"
    return "(" + ", ".join(str(dim) for dim in shape) + ")"


def print_model_summary(
    config: AttrDict | None = None,
    model: nn.Module | None = None,
) -> str:
    """Print a layer table and save it to outputs/metrics/model_summary.txt."""
    cfg = config if config is not None else load_config()
    ensure_dirs(cfg)
    if model is None:
        model = DistilBertClassifier(cfg)
    text = _summary_table(model, cfg)
    out_path = Path(cfg.paths.metrics_dir) / "model_summary.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    logger = get_logger(__name__)
    logger.info("Wrote model summary -> %s", out_path)
    print(text, end="")
    print(f"wrote {out_path}")
    return text


def main(argv: list[str] | None = None) -> None:
    """Instantiate DistilBertClassifier and write the layer summary."""
    del argv
    cfg = load_config()
    set_seed(int(cfg.seed))
    print_model_summary(cfg)


if __name__ == "__main__":
    main()
