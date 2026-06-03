#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whisper_tw.runtime_env import configure_runtime_environment

configure_runtime_environment()

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from whisper_tw.config import load_config, resolve_common_voice_split_source
from whisper_tw.contrastive_router import (
    ContrastiveAdapterRouter,
    ContrastiveRouterSpec,
)
from whisper_tw.data import load_audio_waveform, read_common_voice_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="訓練對比式鑰匙查詢路由。")
    parser.add_argument("--config", default="configs/config.yaml", help="訓練設定檔。")
    parser.add_argument("--max-train-samples", type=int, help="覆寫訓練樣本上限。")
    parser.add_argument("--max-eval-samples", type=int, help="覆寫驗證樣本上限。")
    parser.add_argument("--num-epochs", type=float, help="覆寫訓練週期數。")
    return parser.parse_args()


def get_train_config(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("whisper_train")
    if isinstance(section, dict):
        return section
    return {}


def resolve_split_source(data_cfg: dict[str, Any], split: str) -> str | Path:
    if split == str(data_cfg.get("train_split", "train")) and data_cfg.get("train_tsv"):
        return data_cfg["train_tsv"]
    if split == str(data_cfg.get("eval_split", "dev")) and data_cfg.get("eval_tsv"):
        return data_cfg["eval_tsv"]
    if split == str(data_cfg.get("test_split", "test")) and data_cfg.get("test_tsv"):
        return data_cfg["test_tsv"]
    return resolve_common_voice_split_source(data_cfg, split)


class RouterDataset(Dataset):
    def __init__(
        self,
        *,
        data_cfg: dict[str, Any],
        split: str,
        processor,
        labels: list[str],
        max_samples: int | None,
    ) -> None:
        self.label_to_id = {label: index for index, label in enumerate(labels)}
        split_source = resolve_split_source(data_cfg, split)
        samples = read_common_voice_split(data_cfg.get("root", "data"), split_source)
        self.samples = [
            sample for sample in samples if sample.language_label in self.label_to_id
        ]
        if max_samples is not None:
            self.samples = self.samples[: max(0, int(max_samples))]
        self.processor = processor
        self.sample_rate = int(data_cfg.get("sample_rate", 16000))
        self.max_audio_samples = int(
            self.sample_rate * float(data_cfg.get("max_audio_seconds", 30.0))
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        waveform = load_audio_waveform(sample.audio_path, self.sample_rate)
        waveform = waveform[: self.max_audio_samples]
        features = self.processor.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.sample_rate,
        ).input_features[0]
        return {
            "input_features": features,
            "labels": self.label_to_id[sample.language_label],
        }


class RouterCollator:
    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(
            [int(feature["labels"]) for feature in features],
            dtype=torch.long,
        )
        return batch


def build_autocast(device: torch.device, cfg: dict[str, Any]):
    if device.type != "cuda":
        return nullcontext()
    if bool(cfg.get("bf16", False)):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if bool(cfg.get("fp16", False)):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def compute_metrics(
    predictions: list[int],
    references: list[int],
    *,
    num_labels: int,
) -> dict[str, Any]:
    total = len(references)
    correct = sum(int(pred == ref) for pred, ref in zip(predictions, references))
    confusion = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for pred, ref in zip(predictions, references):
        confusion[ref][pred] += 1

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    f1_scores: list[float] = []
    for label_id in range(num_labels):
        tp = confusion[label_id][label_id]
        fp = sum(confusion[row][label_id] for row in range(num_labels) if row != label_id)
        fn = sum(confusion[label_id][col] for col in range(num_labels) if col != label_id)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": correct / max(1, total),
        "macro_precision": sum(precision_scores) / max(1, len(precision_scores)),
        "macro_recall": sum(recall_scores) / max(1, len(recall_scores)),
        "macro_f1": sum(f1_scores) / max(1, len(f1_scores)),
        "per_label_precision": precision_scores,
        "per_label_recall": recall_scores,
        "per_label_f1": f1_scores,
        "confusion_matrix": confusion,
    }


def evaluate(
    *,
    encoder,
    router: ContrastiveAdapterRouter,
    loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    router.eval()
    losses: list[float] = []
    predictions: list[int] = []
    references: list[int] = []
    positive_similarities: list[float] = []
    negative_similarities: list[float] = []
    progress = tqdm(
        loader,
        desc="eval contrastive router",
        disable=bool(cfg.get("disable_tqdm", False)),
    )
    with torch.no_grad():
        for batch in progress:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            with build_autocast(device, cfg):
                hidden = encoder(input_features).last_hidden_state
            loss, outputs = router.compute_loss(hidden.float(), labels)
            logits = outputs["logits"]
            similarities = outputs["queries"] @ outputs["keys"].transpose(0, 1)
            losses.append(float(loss.detach().cpu()))
            predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
            references.extend(labels.detach().cpu().tolist())

            row_ids = torch.arange(labels.size(0), device=device)
            positive = similarities[row_ids, labels]
            mask = torch.ones_like(similarities, dtype=torch.bool)
            mask[row_ids, labels] = False
            negative = similarities.masked_fill(~mask, float("-inf")).max(dim=-1).values
            positive_similarities.extend(positive.detach().cpu().tolist())
            negative_similarities.extend(negative.detach().cpu().tolist())

    metrics = compute_metrics(
        predictions,
        references,
        num_labels=len(router.spec.labels),
    )
    metrics["loss"] = sum(losses) / max(1, len(losses))
    avg_positive = sum(positive_similarities) / max(1, len(positive_similarities))
    avg_negative = sum(negative_similarities) / max(1, len(negative_similarities))
    metrics["avg_positive_similarity"] = avg_positive
    metrics["avg_max_negative_similarity"] = avg_negative
    metrics["avg_similarity_gap"] = avg_positive - avg_negative
    return metrics


def init_wandb(cfg: dict[str, Any], output_dir: Path):
    report_to = cfg.get("report_to", [])
    if isinstance(report_to, str):
        enabled = report_to.lower() == "wandb"
    else:
        enabled = "wandb" in [str(item).lower() for item in report_to]
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        return None

    os.environ.setdefault("WANDB_PROJECT", str(cfg.get("wandb_project") or "whisper-tw"))
    os.environ.setdefault("WANDB_NAME", str(cfg.get("run_name") or output_dir.name))
    os.environ["WANDB_DISABLE_STATS"] = "false"
    return wandb.init(
        project=os.environ["WANDB_PROJECT"],
        name=os.environ["WANDB_NAME"],
        config=cfg,
    )


def filter_wandb_scalars(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and "confusion_matrix" not in key
    }


def main() -> None:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    args = parse_args()
    print(f"啟動對比式路由訓練: config={args.config}", flush=True)
    config = load_config(args.config)
    train_cfg = get_train_config(config)
    model_cfg = train_cfg.get("model", {}) or {}
    data_cfg = train_cfg.get("data", {}) or {}
    router_cfg = train_cfg.get("contrastive_router", {}) or {}
    if not bool(router_cfg.get("enabled", True)):
        raise ValueError("contrastive_router.enabled 已關閉。")

    labels = [str(label) for label in router_cfg.get("labels", ["zh-TW", "nan-tw"])]
    model_name_or_path = str(
        model_cfg.get("model_name_or_path") or "openai/whisper-medium"
    )
    output_dir = Path(str(router_cfg.get("output_dir") or "artifacts/models/contrastive_router"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bool(router_cfg.get("tf32", False)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"載入 processor: {model_name_or_path}", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        language=str(model_cfg.get("language") or "zh"),
        task=str(model_cfg.get("task") or "transcribe"),
    )
    print(f"載入 Whisper 模型: {model_name_or_path}", flush=True)
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path).to(device)
    print("Whisper 模型載入完成。", flush=True)
    whisper.eval()
    for param in whisper.parameters():
        param.requires_grad = False
    encoder = whisper.model.encoder

    hidden_size = int(getattr(whisper.config, "d_model", 1024))
    spec = ContrastiveRouterSpec(
        labels=tuple(labels),
        hidden_size=hidden_size,
        pooling=str(router_cfg.get("pooling") or "attention"),
        attention_hidden_size=int(router_cfg.get("attention_hidden_size", 256)),
        embedding_size=int(router_cfg.get("embedding_size", 256)),
        hidden_ratio=float(router_cfg.get("hidden_ratio", 0.5)),
        dropout=float(router_cfg.get("dropout", 0.1)),
        temperature=float(router_cfg.get("temperature", 0.07)),
        label_smoothing=float(router_cfg.get("label_smoothing", 0.0)),
        margin=float(router_cfg.get("margin", 0.0)),
        margin_loss_weight=float(router_cfg.get("margin_loss_weight", 0.0)),
    )
    router = ContrastiveAdapterRouter(spec).to(device)

    max_train_samples = (
        args.max_train_samples
        if args.max_train_samples is not None
        else router_cfg.get("max_train_samples")
    )
    max_eval_samples = (
        args.max_eval_samples
        if args.max_eval_samples is not None
        else router_cfg.get("max_eval_samples")
    )
    train_dataset = RouterDataset(
        data_cfg=data_cfg,
        split=str(router_cfg.get("train_split") or data_cfg.get("train_split", "train")),
        processor=processor,
        labels=labels,
        max_samples=None if max_train_samples is None else int(max_train_samples),
    )
    eval_dataset = RouterDataset(
        data_cfg=data_cfg,
        split=str(router_cfg.get("eval_split") or data_cfg.get("eval_split", "dev")),
        processor=processor,
        labels=labels,
        max_samples=None if max_eval_samples is None else int(max_eval_samples),
    )
    if len(train_dataset) == 0:
        raise ValueError("對比式路由訓練資料為空，請先檢查前處理後的 language_label。")
    if len(eval_dataset) == 0:
        raise ValueError("對比式路由驗證資料為空，請先檢查前處理後的 language_label。")

    collator = RouterCollator(processor)
    num_workers = int(router_cfg.get("dataloader_num_workers", 1))
    loader_kwargs = {
        "collate_fn": collator,
        "num_workers": num_workers,
        "pin_memory": bool(router_cfg.get("dataloader_pin_memory", device.type == "cuda")),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            router_cfg.get("dataloader_persistent_workers", False)
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(router_cfg.get("per_device_train_batch_size", 4)),
        shuffle=True,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(router_cfg.get("per_device_eval_batch_size", 4)),
        shuffle=False,
        **loader_kwargs,
    )
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=float(router_cfg.get("learning_rate", 1.0e-4)),
        weight_decay=float(router_cfg.get("weight_decay", 0.01)),
    )
    epochs = float(args.num_epochs if args.num_epochs is not None else router_cfg.get("num_train_epochs", 10.0))
    update_steps = max(1, int(len(train_loader) * epochs))
    warmup_steps = int(router_cfg.get("warmup_steps", 100))
    early_stopping_cfg = dict(router_cfg.get("early_stopping") or {})
    early_stopping_enabled = bool(early_stopping_cfg.get("enabled", True))
    early_stopping_patience = max(1, int(early_stopping_cfg.get("patience", 3)))
    early_stopping_min_delta = float(early_stopping_cfg.get("min_delta", 0.0))
    early_stopping_metric = str(early_stopping_cfg.get("metric") or "macro_f1")
    if early_stopping_metric != "macro_f1":
        raise ValueError("對比式路由早停目前只支援 macro_f1。")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        remaining = max(1, update_steps - warmup_steps)
        return max(0.0, (update_steps - step) / remaining)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    wandb_run = init_wandb(router_cfg, output_dir)

    print(
        json.dumps(
            {
                "config": args.config,
                "model_name_or_path": model_name_or_path,
                "labels": labels,
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
                "output_dir": str(output_dir),
                "device": str(device),
                "router": {
                    "pooling": spec.pooling,
                    "attention_hidden_size": spec.attention_hidden_size,
                    "embedding_size": spec.embedding_size,
                    "hidden_ratio": spec.hidden_ratio,
                    "dropout": spec.dropout,
                    "temperature": spec.temperature,
                    "label_smoothing": spec.label_smoothing,
                    "margin": spec.margin,
                    "margin_loss_weight": spec.margin_loss_weight,
                },
                "early_stopping": {
                    "enabled": early_stopping_enabled,
                    "patience": early_stopping_patience,
                    "min_delta": early_stopping_min_delta,
                    "metric": early_stopping_metric,
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    best_macro_f1 = -1.0
    global_step = 0
    stale_epochs = 0
    full_epochs = max(1, int(epochs))
    best_checkpoint_path = output_dir / "contrastive_router.pt"
    for epoch in range(1, full_epochs + 1):
        router.train()
        losses: list[float] = []
        progress = tqdm(
            train_loader,
            desc=f"train contrastive router {epoch}/{full_epochs}",
            disable=bool(router_cfg.get("disable_tqdm", False)),
        )
        for batch in progress:
            global_step += 1
            input_features = batch["input_features"].to(device)
            labels_tensor = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                with build_autocast(device, router_cfg):
                    hidden = encoder(input_features).last_hidden_state
            loss, _outputs = router.compute_loss(hidden.float(), labels_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                router.parameters(),
                float(router_cfg.get("max_grad_norm", 1.0)),
            )
            optimizer.step()
            scheduler.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            progress.set_postfix(loss=f"{loss_value:.4f}")
            if (
                bool(router_cfg.get("log_step_loss", False))
                and wandb_run is not None
                and global_step % int(router_cfg.get("logging_steps", 25)) == 0
            ):
                wandb_run.log(
                    {
                        "train_step_loss": loss_value,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        eval_metrics = evaluate(
            encoder=encoder,
            router=router,
            loader=eval_loader,
            device=device,
            cfg=router_cfg,
        )
        train_loss = sum(losses) / max(1, len(losses))
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            **{f"eval_{key}": value for key, value in eval_metrics.items()},
        }
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        current_macro_f1 = float(eval_metrics["macro_f1"])
        improved = current_macro_f1 > best_macro_f1 + early_stopping_min_delta
        if wandb_run is not None:
            wandb_run.log(
                filter_wandb_scalars(metrics),
                step=global_step,
            )

        if improved:
            best_macro_f1 = current_macro_f1
            stale_epochs = 0
            payload = router.checkpoint_payload()
            payload.update(
                {
                    "model_name_or_path": model_name_or_path,
                    "best_macro_f1": best_macro_f1,
                    "epoch": epoch,
                }
            )
            torch.save(payload, output_dir / "contrastive_router.pt")
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            stale_epochs += 1

        if early_stopping_enabled and stale_epochs >= early_stopping_patience:
            print(
                json.dumps(
                    {
                        "stage": "early_stopping",
                        "epoch": epoch,
                        "metric": early_stopping_metric,
                        "best_macro_f1": best_macro_f1,
                        "stale_epochs": stale_epochs,
                        "patience": early_stopping_patience,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break

    if wandb_run is not None:
        wandb_run.finish()
    print(f"已保存最佳對比式路由: {output_dir / 'contrastive_router.pt'}", flush=True)


if __name__ == "__main__":
    main()
