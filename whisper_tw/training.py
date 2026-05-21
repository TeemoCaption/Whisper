from __future__ import annotations

from pathlib import Path
from typing import Any

import os
import tempfile
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import WhisperFeatureExtractor

from .bopomofo import BopomofoVocab
from .config import resolve_device
from .data import CommonVoiceTaiwanDataset, WhisperTWCollator, build_audio_augmentor
from .model import WhisperTWModel
from .text_normalization import build_text_normalizer
from .tokenizer import SentencePieceTextTokenizer


class WandbLogger:
    def __init__(self, config: dict[str, Any]) -> None:
        wandb_cfg = config.get("wandb", {})
        self.enabled = bool(wandb_cfg.get("enabled", False))
        self.run = None
        if not self.enabled:
            return

        import wandb

        mode = os.environ.get("WANDB_MODE") or str(wandb_cfg.get("mode", "online"))
        default_wandb_root = Path(tempfile.gettempdir()) / "wandb" / "Whisper-TW"
        wandb_root = Path(
            os.environ.get("WANDB_DIR")
            or wandb_cfg.get("dir")
            or default_wandb_root / "run"
        )
        wandb_config_dir = Path(
            os.environ.get("WANDB_CONFIG_DIR")
            or wandb_cfg.get("config_dir")
            or default_wandb_root / "config"
        )
        wandb_cache_dir = Path(
            os.environ.get("WANDB_CACHE_DIR")
            or wandb_cfg.get("cache_dir")
            or default_wandb_root / "cache"
        )
        wandb_data_dir = Path(
            os.environ.get("WANDB_DATA_DIR")
            or wandb_cfg.get("data_dir")
            or default_wandb_root / "data"
        )
        for path in (wandb_root, wandb_config_dir, wandb_cache_dir, wandb_data_dir):
            path.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("WANDB_DIR", str(wandb_root))
        os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_config_dir))
        os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_cache_dir))
        os.environ.setdefault("WANDB_DATA_DIR", str(wandb_data_dir))

        if mode == "online":
            api_key = os.environ.get("WANDB_API_KEY", "").strip()
            netrc_candidates = [
                Path.home() / ".netrc",
                Path.home() / "_netrc",
            ]
            if os.name == "nt":
                netrc_candidates.extend(
                    [
                        Path.home().parent / "_netrc",
                        Path.home().parent / ".netrc",
                    ]
                )
            has_netrc = any(path.exists() for path in netrc_candidates)
            if not api_key and not has_netrc:
                raise RuntimeError(
                    "wandb 已啟用為 online，但目前沒有可用的 WANDB_API_KEY。"
                    "請先執行 `wandb login`，或先設定 `WANDB_API_KEY`。"
                )

        settings = wandb.Settings(
            console=str(wandb_cfg.get("console", "redirect")),
            x_files_dir=str(wandb_root),
        )
        self.run = wandb.init(
            project=str(wandb_cfg.get("project", "Whisper-TW")),
            name=wandb_cfg.get("name"),
            mode=mode,
            config=config,
            dir=str(wandb_root),
            settings=settings,
        )
        self.run.define_metric("epoch")
        self.run.define_metric("train_*", step_metric="epoch")
        self.run.define_metric("val_*", step_metric="epoch")
        self.run.define_metric("best_*", step_metric="epoch")
        self.run.define_metric("early_stopping_*", step_metric="epoch")
        self.run.define_metric("learning_rate", step_metric="epoch")
        self.run.define_metric("mixed_precision", step_metric="epoch")

    def log(self, metrics: dict[str, float | int]) -> None:
        if not self.enabled:
            return
        import wandb

        epoch = metrics.get("epoch")
        wandb.log(metrics, step=int(epoch) if epoch is not None else None)

    def log_message(self, message: str, epoch: int | None = None) -> None:
        if not self.enabled:
            return
        import wandb

        prefix: list[str] = []
        if epoch is not None:
            prefix.append(f"epoch={epoch}")
        payload = f"[{', '.join(prefix)}] {message}" if prefix else message
        metrics: dict[str, Any] = {"log_message": payload}
        if epoch is not None:
            metrics["epoch"] = epoch
        wandb.log(metrics, step=epoch)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def build_components(
    config: dict[str, Any], split: str, max_samples: int | None = None
):
    data_cfg = config["data"]
    train_cfg = config.get("training", {})
    feature_cache_cfg = data_cfg.get("feature_cache", {})
    feature_cache_enabled = bool(feature_cache_cfg.get("enabled", False))
    feature_cache_dir = feature_cache_cfg.get("root")
    train_split = data_cfg.get("train_split", "train")
    feature_cache_variants = int(
        feature_cache_cfg.get(
            "train_variants",
            1,
        )
    )
    tokenizer = SentencePieceTextTokenizer(config["tokenizer"]["model_path"])
    bopomofo_vocab = BopomofoVocab.default()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config["model"]["whisper_name"]
    )
    dataset = CommonVoiceTaiwanDataset(
        data_root=data_cfg["root"],
        split=split,
        text_tokenizer=tokenizer,
        bopomofo_vocab=bopomofo_vocab,
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        max_audio_seconds=float(data_cfg.get("max_audio_seconds", 30.0)),
        max_samples=max_samples,
        text_normalizer=build_text_normalizer(data_cfg.get("text_normalization")),
        audio_augmentor=(
            build_audio_augmentor(
                sample_rate=int(data_cfg.get("sample_rate", 16000)),
                config=data_cfg.get("audio_augmentation"),
            )
            if split == train_split
            and not feature_cache_enabled
            else None
        ),
        feature_cache_dir=feature_cache_dir if feature_cache_enabled else None,
        require_feature_cache=bool(feature_cache_cfg.get("strict", True)),
        feature_cache_variants=(
            feature_cache_variants if split == train_split else 1
        ),
        sample_cached_variant=feature_cache_enabled and split == train_split,
    )
    collator = WhisperTWCollator(
        feature_extractor=feature_extractor,
        text_pad_id=tokenizer.pad_id,
        bopomofo_pad_id=bopomofo_vocab.pad_id,
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        feature_padding=str(train_cfg.get("feature_padding", "max_length")),
        feature_pad_to_multiple_of=(
            int(train_cfg["feature_pad_to_multiple_of"])
            if train_cfg.get("feature_pad_to_multiple_of") is not None
            else None
        ),
    )
    model = WhisperTWModel(
        config=config,
        vocab_size=tokenizer.vocab_size,
        bopomofo_vocab_size=bopomofo_vocab.size,
        text_pad_id=tokenizer.pad_id,
    )
    return tokenizer, dataset, collator, model


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def build_dataloader_kwargs(
    train_cfg: dict[str, Any], device: torch.device, split: str = "train"
) -> dict[str, Any]:
    if split == "eval":
        num_workers = int(train_cfg.get("eval_num_workers", 0))
        pin_memory = bool(
            train_cfg.get(
                "eval_pin_memory", train_cfg.get("pin_memory", device.type == "cuda")
            )
        )
        persistent_workers = bool(train_cfg.get("eval_persistent_workers", False))
        prefetch_factor = int(train_cfg.get("eval_prefetch_factor", 2))
    else:
        num_workers = int(train_cfg.get("num_workers", 4))
        pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda"))
        persistent_workers = bool(train_cfg.get("persistent_workers", num_workers > 0))
        prefetch_factor = int(train_cfg.get("prefetch_factor", 2))

    dataloader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    return dataloader_kwargs


def use_mixed_precision(train_cfg: dict[str, Any], device: torch.device) -> bool:
    return bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"


def get_amp_dtype(
    train_cfg: dict[str, Any], device: torch.device
) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    amp_dtype = str(train_cfg.get("amp_dtype", "auto")).lower()
    if amp_dtype == "original":
        return None
    if amp_dtype in {"auto", "default"}:
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if amp_dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if amp_dtype in {"fp16", "float16", "half"}:
        return torch.float16
    if amp_dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")


def use_grad_scaler(amp_enabled: bool, amp_dtype: torch.dtype | None) -> bool:
    return amp_enabled and amp_dtype in {None, torch.float16}


def configure_multiprocessing(train_cfg: dict[str, Any]) -> None:
    sharing_strategy = str(train_cfg.get("sharing_strategy", "file_system"))
    current_strategy = mp.get_sharing_strategy()
    if sharing_strategy != current_strategy:
        mp.set_sharing_strategy(sharing_strategy)


def configure_training_runtime(train_cfg: dict[str, Any], device: torch.device) -> None:
    configure_multiprocessing(train_cfg)
    if device.type != "cuda":
        return
    allow_tf32 = bool(train_cfg.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", True))
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(
            bool(train_cfg.get("enable_flash_sdp", True))
        )
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(
            bool(train_cfg.get("enable_mem_efficient_sdp", True))
        )
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(
            bool(train_cfg.get("enable_math_sdp", True))
        )
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(
            str(train_cfg.get("float32_matmul_precision", "high"))
        )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
    if scheduler_type in {"none", "off", "disabled"}:
        return None
    if scheduler_type not in {"reduce_on_plateau", "warmup_reduce_on_plateau"}:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 2)),
        threshold=float(scheduler_cfg.get("threshold", 0.0)),
        threshold_mode=str(scheduler_cfg.get("threshold_mode", "rel")),
        cooldown=int(scheduler_cfg.get("cooldown", 0)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        eps=float(scheduler_cfg.get("eps", 1e-8)),
    )


class WarmupReduceOnPlateau:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        plateau_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
        warmup_epochs: int,
        warmup_start_factor: float,
    ) -> None:
        self.optimizer = optimizer
        self.plateau_scheduler = plateau_scheduler
        self.warmup_epochs = max(warmup_epochs, 0)
        self.warmup_start_factor = max(min(warmup_start_factor, 1.0), 0.0)
        for group in self.optimizer.param_groups:
            group["target_lr"] = float(group["lr"])
            if self.warmup_epochs > 0:
                group["lr"] = float(group["target_lr"]) * self.warmup_start_factor

    def step(self, metric: float | None, epoch: int) -> None:
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            factor = self.warmup_start_factor + (
                1.0 - self.warmup_start_factor
            ) * progress
            for group in self.optimizer.param_groups:
                group["lr"] = float(group["target_lr"]) * factor
            return
        if self.plateau_scheduler is not None and metric is not None:
            self.plateau_scheduler.step(metric)


def build_lr_controller(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
) -> WarmupReduceOnPlateau | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
    if scheduler_type != "warmup_reduce_on_plateau":
        return build_scheduler(optimizer, train_cfg)
    plateau_scheduler = build_scheduler(optimizer, train_cfg)
    return WarmupReduceOnPlateau(
        optimizer=optimizer,
        plateau_scheduler=plateau_scheduler,
        warmup_epochs=int(scheduler_cfg.get("warmup_epochs", 3)),
        warmup_start_factor=float(scheduler_cfg.get("warmup_start_factor", 0.1)),
    )


def get_validation_monitor_value(
    val_metrics: dict[str, float],
    monitor_name: str,
) -> float:
    metric_aliases = {
        "val_loss": "loss",
        "loss": "loss",
        "val_text_loss": "text_loss",
        "text_loss": "text_loss",
        "val_bopomofo_ctc_loss": "bopomofo_ctc_loss",
        "bopomofo_ctc_loss": "bopomofo_ctc_loss",
    }
    metric_key = metric_aliases.get(monitor_name)
    if metric_key is None:
        raise ValueError(f"Unsupported validation monitor: {monitor_name}")
    return float(val_metrics[metric_key])


def build_optimizer(
    model: WhisperTWModel,
    train_cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    base_lr = float(train_cfg["learning_rate"])
    encoder_lr = float(train_cfg.get("encoder_learning_rate", base_lr))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    encoder_params: list[torch.nn.Parameter] = []
    task_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("whisper.encoder."):
            encoder_params.append(param)
        else:
            task_params.append(param)

    param_groups: list[dict[str, Any]] = []
    if task_params:
        param_groups.append(
            {"params": task_params, "lr": base_lr, "weight_decay": weight_decay}
        )
    if encoder_params:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            }
        )
    optimizer_kwargs: dict[str, Any] = {}
    if bool(train_cfg.get("fused_adamw", True)) and torch.cuda.is_available():
        optimizer_kwargs["fused"] = True
    return torch.optim.AdamW(param_groups, **optimizer_kwargs)


def maybe_compile_model(
    model: WhisperTWModel,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> WhisperTWModel:
    compile_cfg = train_cfg.get("compile", {})
    if device.type != "cuda":
        return model
    if not bool(compile_cfg.get("enabled", False)):
        return model
    if not hasattr(torch, "compile"):
        return model
    return torch.compile(
        model,
        mode=str(compile_cfg.get("mode", "reduce-overhead")),
        fullgraph=bool(compile_cfg.get("fullgraph", False)),
        dynamic=bool(compile_cfg.get("dynamic", True)),
    )


def unwrap_model(model: WhisperTWModel) -> WhisperTWModel:
    return getattr(model, "_orig_mod", model)


def autocast_context(
    device: torch.device, amp_enabled: bool, amp_dtype: torch.dtype | None
):
    device_type = "cuda" if device.type == "cuda" else device.type
    autocast_kwargs: dict[str, Any] = {
        "device_type": device_type,
        "enabled": amp_enabled,
    }
    if amp_dtype is not None:
        autocast_kwargs["dtype"] = amp_dtype
    return torch.amp.autocast(**autocast_kwargs)


def shutdown_dataloader_workers(dataloader: DataLoader) -> None:
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is None:
        return
    shutdown_workers = getattr(iterator, "_shutdown_workers", None)
    if shutdown_workers is not None:
        shutdown_workers()
    dataloader._iterator = None


def shutdown_nonpersistent_dataloader_workers(dataloader: DataLoader) -> None:
    if bool(getattr(dataloader, "persistent_workers", False)):
        return
    shutdown_dataloader_workers(dataloader)


@torch.no_grad()
def val_loss(
    model: WhisperTWModel,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_text_loss = 0.0
    total_ctc_loss = 0.0
    total_batches = 0

    progress = tqdm(
        dataloader,
        desc="val",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, amp_enabled, amp_dtype):
            output = model(
                input_features=batch["input_features"],
                decoder_input_ids=batch["decoder_input_ids"],
                labels=batch["labels"],
                bopomofo_labels=batch["bopomofo_labels"],
                bopomofo_label_lengths=batch["bopomofo_label_lengths"],
            )
        if output.loss is None:
            raise RuntimeError("Validation loss is None.")
        total_loss += float(output.loss.detach().cpu())
        total_text_loss += (
            float(output.text_loss.detach().cpu())
            if output.text_loss is not None
            else 0.0
        )
        total_ctc_loss += (
            float(output.bopomofo_ctc_loss.detach().cpu())
            if output.bopomofo_ctc_loss is not None
            else 0.0
        )
        total_batches += 1
        progress.set_postfix(loss=f"{total_loss / total_batches:.4f}")

    model.train()
    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "text_loss": total_text_loss / denom,
        "bopomofo_ctc_loss": total_ctc_loss / denom,
    }


def train(config: dict[str, Any], max_samples: int | None = None) -> Path:
    train_cfg = config["training"]
    logger = WandbLogger(config)
    device = torch.device(resolve_device(config))
    configure_training_runtime(train_cfg, device)
    try:
        _, dataset, collator, model = build_components(
            config, config["data"].get("train_split", "train"), max_samples
        )
        val_dataset = CommonVoiceTaiwanDataset(
            data_root=config["data"]["root"],
            split=config["data"].get("dev_split", "dev"),
            text_tokenizer=SentencePieceTextTokenizer(
                config["tokenizer"]["model_path"]
            ),
            bopomofo_vocab=BopomofoVocab.default(),
            sample_rate=int(config["data"].get("sample_rate", 16000)),
            max_audio_seconds=float(config["data"].get("max_audio_seconds", 30.0)),
            max_samples=max_samples,
            text_normalizer=build_text_normalizer(
                config["data"].get("text_normalization")
            ),
            feature_cache_dir=(
                config["data"].get("feature_cache", {}).get("root")
                if bool(config["data"].get("feature_cache", {}).get("enabled", False))
                else None
            ),
            require_feature_cache=bool(
                config["data"].get("feature_cache", {}).get("strict", True)
            ),
            feature_cache_variants=1,
            sample_cached_variant=False,
        )
        model.to(device)
        model = maybe_compile_model(model, train_cfg, device)
        model.train()
        amp_enabled = use_mixed_precision(train_cfg, device)
        amp_dtype = get_amp_dtype(train_cfg, device)
        train_dataloader_kwargs = build_dataloader_kwargs(
            train_cfg, device, split="train"
        )
        val_dataloader_kwargs = build_dataloader_kwargs(train_cfg, device, split="eval")

        train_dataloader = DataLoader(
            dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            collate_fn=collator,
            **train_dataloader_kwargs,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])),
            shuffle=False,
            collate_fn=collator,
            **val_dataloader_kwargs,
        )
        optimizer = build_optimizer(model, train_cfg)
        scheduler = build_lr_controller(optimizer, train_cfg)
        scaler = torch.amp.GradScaler(
            device=device.type,
            enabled=use_grad_scaler(amp_enabled, amp_dtype),
        )

        early_cfg = train_cfg.get("early_stopping", {})
        early_enabled = bool(early_cfg.get("enabled", True))
        patience = int(early_cfg.get("patience", 5))
        min_delta = float(early_cfg.get("min_delta", 0.001))
        scheduler_cfg = train_cfg.get("scheduler", {})
        scheduler_monitor = str(scheduler_cfg.get("monitor", "val_loss"))
        early_monitor = str(early_cfg.get("monitor", scheduler_monitor))
        checkpoint_monitor = str(
            train_cfg.get("checkpoint_monitor", early_monitor)
        )
        epochs_without_improvement = 0
        best_early_monitor_value = float("inf")
        best_checkpoint_monitor_value = float("inf")
        best_checkpoint: Path | None = None
        gradient_accumulation_steps = max(
            int(train_cfg.get("gradient_accumulation_steps", 1)),
            1,
        )
        num_epochs = int(train_cfg.get("num_epochs", 1))
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            epoch_text_loss = 0.0
            epoch_ctc_loss = 0.0
            epoch_batches = 0
            optimizer.zero_grad(set_to_none=True)
            phase_message = f"epoch={epoch} phase=train"
            print(phase_message)
            logger.log_message(phase_message, epoch=epoch)
            train_progress = tqdm(
                train_dataloader,
                desc=f"train {epoch}/{int(train_cfg.get('num_epochs', 1))}",
                leave=False,
                dynamic_ncols=True,
            )
            total_train_batches = len(train_dataloader)
            log_every = max(int(train_cfg.get("log_every", 10)), 1)
            for batch_index, batch in enumerate(train_progress, start=1):
                batch = move_batch_to_device(batch, device)
                with autocast_context(device, amp_enabled, amp_dtype):
                    output = model(
                        input_features=batch["input_features"],
                        decoder_input_ids=batch["decoder_input_ids"],
                        labels=batch["labels"],
                        bopomofo_labels=batch["bopomofo_labels"],
                        bopomofo_label_lengths=batch["bopomofo_label_lengths"],
                    )
                if output.loss is None:
                    raise RuntimeError("Training loss is None.")
                scaled_loss = output.loss / gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                should_step = (
                    batch_index % gradient_accumulation_steps == 0
                    or batch_index == total_train_batches
                )
                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += float(output.loss.detach().cpu())
                epoch_batches += 1
                text_loss = (
                    output.text_loss.item()
                    if output.text_loss is not None
                    else float("nan")
                )
                ctc_loss = (
                    output.bopomofo_ctc_loss.item()
                    if output.bopomofo_ctc_loss is not None
                    else float("nan")
                )
                epoch_text_loss += float(text_loss)
                epoch_ctc_loss += float(ctc_loss)
                if batch_index == 1 or batch_index % log_every == 0:
                    logger.log_message(
                        (
                            f"phase=train batch={batch_index}/{total_train_batches} "
                            f"progress={batch_index / max(total_train_batches, 1) * 100.0:.2f}% "
                            f"loss={float(output.loss.detach().cpu()):.4f} "
                            f"text_loss={float(text_loss):.4f} "
                            f"bopomofo_ctc_loss={float(ctc_loss):.4f}"
                        ),
                        epoch=epoch,
                    )
                train_progress.set_postfix(
                    loss=f"{output.loss.item():.4f}",
                    accum=f"{(batch_index - 1) % gradient_accumulation_steps + 1}/{gradient_accumulation_steps}",
                    early_stop=(
                        f"{epochs_without_improvement}/{patience}"
                        if early_enabled
                        else "off"
                    ),
                )
            train_loss = epoch_loss / max(epoch_batches, 1)
            train_text_loss = epoch_text_loss / max(epoch_batches, 1)
            train_ctc_loss = epoch_ctc_loss / max(epoch_batches, 1)
            shutdown_nonpersistent_dataloader_workers(train_dataloader)
            val_metrics = val_loss(
                model,
                val_dataloader,
                device,
                amp_enabled,
                amp_dtype,
            )
            shutdown_nonpersistent_dataloader_workers(val_dataloader)
            scheduler_metric = get_validation_monitor_value(
                val_metrics,
                scheduler_monitor,
            )
            early_metric = get_validation_monitor_value(val_metrics, early_monitor)
            checkpoint_metric = get_validation_monitor_value(
                val_metrics,
                checkpoint_monitor,
            )
            if scheduler is not None:
                if isinstance(scheduler, WarmupReduceOnPlateau):
                    scheduler.step(scheduler_metric, epoch)
                else:
                    scheduler.step(scheduler_metric)
            current_lr = float(optimizer.param_groups[0]["lr"])
            current_encoder_lr = (
                float(optimizer.param_groups[1]["lr"])
                if len(optimizer.param_groups) > 1
                else current_lr
            )
            scheduler_phase = (
                "warmup"
                if isinstance(scheduler, WarmupReduceOnPlateau)
                and scheduler.warmup_epochs > 0
                and epoch <= scheduler.warmup_epochs
                else "plateau"
            )
            epoch_summary_message = (
                f"epoch={epoch} train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_text_loss={val_metrics['text_loss']:.4f} "
                f"val_bopomofo_ctc_loss={val_metrics['bopomofo_ctc_loss']:.4f} "
                f"scheduler_monitor={scheduler_monitor}:{scheduler_metric:.4f} "
                f"early_monitor={early_monitor}:{early_metric:.4f} "
                f"checkpoint_monitor={checkpoint_monitor}:{checkpoint_metric:.4f} "
                f"learning_rate={current_lr:.8f} "
                f"encoder_learning_rate={current_encoder_lr:.8f} "
                f"scheduler_phase={scheduler_phase}"
            )
            print(epoch_summary_message)
            logger.log_message(epoch_summary_message, epoch=epoch)

            logger.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "train_text_loss": train_text_loss,
                    "val_text_loss": val_metrics["text_loss"],
                    "train_bopomofo_ctc_loss": train_ctc_loss,
                    "val_bopomofo_ctc_loss": val_metrics["bopomofo_ctc_loss"],
                    "scheduler_monitor_value": scheduler_metric,
                    "early_monitor_value": early_metric,
                    "checkpoint_monitor_value": checkpoint_metric,
                    "learning_rate": current_lr,
                    "encoder_learning_rate": current_encoder_lr,
                    "scheduler_phase_is_warmup": int(scheduler_phase == "warmup"),
                    "mixed_precision": int(amp_enabled),
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                },
            )

            improved_for_early_stop = early_metric < (
                best_early_monitor_value - min_delta
            )
            improved_for_checkpoint = checkpoint_metric < (
                best_checkpoint_monitor_value - min_delta
            )

            if improved_for_early_stop:
                best_early_monitor_value = early_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                no_improve_message = (
                    f"epoch={epoch} early_stopping_no_improve="
                    f"{epochs_without_improvement}/{patience}"
                )
                print(no_improve_message)
                logger.log_message(no_improve_message, epoch=epoch)
                logger.log(
                    {
                        "epoch": epoch,
                        "early_stopping_no_improve_epochs": epochs_without_improvement,
                        "early_stopping_patience": patience,
                    },
                )

            if improved_for_checkpoint:
                best_checkpoint_monitor_value = checkpoint_metric
                logger.log(
                    {
                        "epoch": epoch,
                        "best_checkpoint_monitor_value": best_checkpoint_monitor_value,
                        "best_epoch": epoch,
                    },
                )
                best_checkpoint = save_checkpoint(
                    model,
                    config,
                    epoch,
                    name="best",
                    metrics={
                        "best_checkpoint_monitor_value": best_checkpoint_monitor_value,
                        "best_early_monitor_value": best_early_monitor_value,
                        "checkpoint_monitor": checkpoint_monitor,
                        **val_metrics,
                    },
                )
                logger.log_message(f"checkpoint={best_checkpoint}", epoch=epoch)

            if early_enabled and epochs_without_improvement >= patience:
                early_stop_message = (
                    f"early_stopping=triggered epoch={epoch} "
                    f"best_{early_monitor}={best_early_monitor_value:.4f}"
                )
                print(early_stop_message)
                logger.log_message(early_stop_message, epoch=epoch)
                logger.log(
                    {
                        "epoch": epoch,
                        "early_stopping_triggered": 1,
                        "early_stopping_stopped_epoch": epoch,
                    },
                )
                break

        final_checkpoint = save_checkpoint(model, config, epoch, name="last")
        logger.log_message(f"checkpoint={final_checkpoint}", epoch=epoch)
        if best_checkpoint is None:
            return final_checkpoint
        return best_checkpoint
    finally:
        logger.finish()


def save_checkpoint(
    model: WhisperTWModel,
    config: dict[str, Any],
    epoch: int,
    name: str = "last",
    metrics: dict[str, float] | None = None,
) -> Path:
    output_dir = Path(config["training"].get("output_dir", "artifacts/checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"whisper_tw_{name}.pt"
    raw_model = unwrap_model(model)
    torch.save(
        {
            "model": raw_model.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": metrics or {},
        },
        checkpoint_path,
    )
    print(f"checkpoint={checkpoint_path}")
    return checkpoint_path
