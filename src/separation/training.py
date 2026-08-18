"""Training entry point for the separation experiment."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from src.separation.datasets import Dataset, Split, build_loader, parse_dataset
from src.separation.models import SeparationConvNet


EXPERIMENT = "separation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run separation pretraining.")
    parser.add_argument(
        "--config",
        default="src/separation/config.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return payload


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if (
            isinstance(value, Mapping)
            and isinstance(base.get(key), dict)
            and "." not in key
        ):
            deep_update(base[key], value)
        elif "." in key:
            set_dotted(base, key, value)
        else:
            base[key] = value
    return base


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ValueError(f"Cannot set dotted config key {dotted_key!r}")
    target[parts[-1]] = value


def init_wandb(config: dict[str, Any], timestamp: str) -> Any | None:
    wandb_config = config.get("wandb", {})
    mode = wandb_config.get("mode", "online")
    if mode == "disabled":
        return None
    if mode not in {"online", "offline"}:
        raise ValueError("wandb.mode must be online, offline, or disabled")

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "wandb is required unless wandb.mode is disabled in the config"
        ) from error

    run = wandb.init(
        project=f"{EXPERIMENT}_{timestamp}",
        entity=wandb_config.get("entity"),
        group=wandb_config.get("group"),
        tags=wandb_config.get("tags") or None,
        mode=mode,
        config=config,
    )
    wandb.define_metric("separation_step")
    wandb.define_metric("regular_step")
    wandb.define_metric("separation loss", step_metric="separation_step")
    wandb.define_metric("regular loss", step_metric="regular_step")
    wandb.define_metric("train/loss", step_metric="regular_step")
    wandb.define_metric("train/accuracy", step_metric="regular_step")
    wandb.define_metric("regular/accuracy", step_metric="regular_step")
    wandb.define_metric("validation/*", step_metric="regular_step")
    return run


def config_with_wandb_overrides(config: dict[str, Any], run: Any | None) -> dict[str, Any]:
    if run is None:
        return config
    updates = dict(run.config)
    updates.pop("sweep", None)
    return deep_update(copy.deepcopy(config), updates)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but CUDA is not available")
    return device


def build_optimizer(
    *,
    parameters: Iterable[torch.nn.Parameter],
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer_type == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer.type: {optimizer_type!r}")


def selected_layers(config: dict[str, Any]) -> list[int]:
    variant = config["variant"]
    variants = config["variants"]
    if variant not in variants:
        choices = ", ".join(sorted(variants))
        raise ValueError(f"Unknown variant {variant!r}; choose one of {choices}")
    layers = list(variants[variant])
    for layer in layers:
        if layer not in (0, 1, 2, 3):
            raise ValueError("Separation layers must be 0, 1, 2, or 3")
    return layers


def separation_loss(
    model: SeparationConvNet,
    images: torch.Tensor,
    layers: list[int],
    pairs_per_layer: int,
) -> tuple[torch.Tensor, float]:
    if not layers:
        raise ValueError("separation_loss requires at least one selected layer")

    inputs = model.conv_inputs(images, detach_layer_inputs=True)
    convolutions = (model.conv1, model.conv2, model.conv3, model.conv4)
    losses = []
    cosine_abs_sum = 0.0
    cosine_count = 0

    for layer_idx in layers:
        layer_input = inputs[layer_idx]
        convolution = convolutions[layer_idx]
        kernel_size = convolution.kernel_size[0]
        height, width = layer_input.shape[-2:]
        if height < kernel_size or width < kernel_size:
            raise ValueError(
                f"Layer {layer_idx} input is too small for a {kernel_size}x{kernel_size} patch"
            )

        patches = F.unfold(layer_input, kernel_size=kernel_size)
        patch_count = patches.shape[-1]
        for image_idx in range(layer_input.shape[0]):
            sample_count = pairs_per_layer * 2
            locations = torch.randint(
                patch_count,
                (sample_count,),
                device=layer_input.device,
            )
            sampled = patches[image_idx, :, locations]
            representations = F.linear(
                sampled.transpose(0, 1),
                convolution.weight.flatten(1),
                convolution.bias,
            )
            for response_a, response_b in representations.reshape(
                pairs_per_layer, 2, -1
            ):
                cosine = F.cosine_similarity(response_a, response_b, dim=0, eps=1e-8)
                losses.append(cosine.square())
                cosine_abs_sum += float(cosine.detach().abs().cpu())
                cosine_count += 1

    return torch.stack(losses).mean(), cosine_abs_sum / max(1, cosine_count)


def run_separation_pretraining(
    *,
    model: SeparationConvNet,
    train_loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    device: torch.device,
    run: Any | None,
) -> None:
    layers = selected_layers(config)
    if not layers:
        return

    separation_config = config["separation"]
    optimizer = build_optimizer(
        parameters=model.parameters(),
        optimizer_type=config["optimizer"]["type"],
        learning_rate=separation_config["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    steps_per_epoch = separation_config.get("steps_per_epoch")
    global_step = 0
    model.train()

    for epoch in range(separation_config["epochs"]):
        for batch_idx, (images, _) in enumerate(train_loader):
            if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                break
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, cosine_abs = separation_loss(
                model=model,
                images=images,
                layers=layers,
                pairs_per_layer=separation_config["pairs_per_layer"],
            )
            loss.backward()
            optimizer.step()

            global_step += 1
            if run is not None:
                run.log(
                    {
                        "separation loss": float(loss.detach().cpu()),
                        "separation/cosine_abs": cosine_abs,
                        "separation/epoch": epoch,
                        "separation_step": global_step,
                        "learning_rate/separation": optimizer.param_groups[0]["lr"],
                    }
                )


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return float((predictions == labels).float().mean().detach().cpu())


def evaluate(
    *,
    model: SeparationConvNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.detach().cpu()) * labels.numel()
            total_correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
            total_seen += labels.numel()
    model.train()
    return total_loss / total_seen, total_correct / total_seen


def run_regular_training(
    *,
    model: SeparationConvNet,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    device: torch.device,
    run: Any | None,
) -> None:
    regular_config = config["regular"]
    optimizer = build_optimizer(
        parameters=model.parameters(),
        optimizer_type=config["optimizer"]["type"],
        learning_rate=regular_config["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    model.train()

    for epoch in range(regular_config["epochs"]):
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            global_step += 1
            if run is not None:
                run.log(
                    {
                        "regular loss": float(loss.detach().cpu()),
                        "train/loss": float(loss.detach().cpu()),
                        "train/accuracy": accuracy(logits, labels),
                        "regular/accuracy": accuracy(logits, labels),
                        "regular/epoch": epoch,
                        "epoch": epoch,
                        "regular_step": global_step,
                        "learning_rate/regular": optimizer.param_groups[0]["lr"],
                    }
                )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            loader=validation_loader,
            device=device,
            criterion=criterion,
        )
        if run is not None:
            run.log(
                {
                    "validation/loss": validation_loss,
                    "validation/accuracy": validation_accuracy,
                    "regular/epoch": epoch,
                    "regular_step": global_step,
                }
            )


def save_checkpoint(
    *,
    model: SeparationConvNet,
    config: dict[str, Any],
    timestamp: str,
) -> None:
    checkpoint_config = config["checkpoint"]
    if not checkpoint_config.get("enabled", True):
        return
    checkpoint_dir = Path(checkpoint_config["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{timestamp}_{config['dataset']}_{config['variant']}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "timestamp": timestamp,
        },
        path,
    )


def resolved_config(config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    parse_dataset(config["dataset"])
    if isinstance(config["variant"], list):
        raise ValueError("Use a single variant per run; sweep over variant for grids")
    config["model"]["channels"] = [int(value) for value in config["model"]["channels"]]
    config["seed"] = int(config["seed"])
    config["dataloader"]["batch_size"] = int(config["dataloader"]["batch_size"])
    config["dataloader"]["num_workers"] = int(config["dataloader"]["num_workers"])
    config["separation"]["epochs"] = int(config["separation"]["epochs"])
    config["separation"]["pairs_per_layer"] = int(
        config["separation"]["pairs_per_layer"]
    )
    if config["separation"]["pairs_per_layer"] < 1:
        raise ValueError("separation.pairs_per_layer must be positive")
    if len(config["model"]["channels"]) != 4:
        raise ValueError("model.channels must contain four channel widths")
    config["regular"]["epochs"] = int(config["regular"]["epochs"])
    selected_layers(config)
    return config


def run_experiment(config: dict[str, Any], timestamp: str, run: Any | None) -> None:
    config = resolved_config(config)
    seed_everything(config["seed"])
    device = select_device(config["device"])
    dataset = parse_dataset(config["dataset"])
    pin_memory = bool(config["dataloader"]["pin_memory"]) and device.type == "cuda"

    train_loader = build_loader(
        data_root=config["data_root"],
        dataset=dataset,
        split=Split.TRAIN,
        batch_size=config["dataloader"]["batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=pin_memory,
        seed=config["seed"],
        normalize=config["dataloader"]["normalize"],
    )
    validation_loader = build_loader(
        data_root=config["data_root"],
        dataset=dataset,
        split=Split.VALIDATION,
        batch_size=config["dataloader"]["batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=pin_memory,
        seed=config["seed"],
        normalize=config["dataloader"]["normalize"],
    )

    model = SeparationConvNet(
        dataset=dataset,
        channels=config["model"]["channels"],
        kernel_size=config["model"]["kernel_size"],
        dropout=config["model"]["dropout"],
    ).to(device)

    if run is not None:
        run.config.update(
            {
                "resolved": json.loads(json.dumps(config)),
                "selected_layers": selected_layers(config),
            },
            allow_val_change=True,
        )

    run_separation_pretraining(
        model=model,
        train_loader=train_loader,
        config=config,
        device=device,
        run=run,
    )
    run_regular_training(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=config,
        device=device,
        run=run,
    )
    save_checkpoint(model=model, config=config, timestamp=timestamp)


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    config = load_yaml(args.config)
    run = init_wandb(config, timestamp)
    try:
        config = config_with_wandb_overrides(config, run)
        run_experiment(config, timestamp, run)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
