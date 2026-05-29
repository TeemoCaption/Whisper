#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whisper_tw.config import load_config, resolve_common_voice_split_source
from whisper_tw.data import load_audio_waveform, read_common_voice_split
from whisper_tw.lang_classifier import (
    LanguageClassifierSpec,
    WhisperLanguageClassifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="訓練 Whisper 編碼器語言分類頭。")
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


class LanguageDataset(Dataset):
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


class LanguageCollator:
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

    f1_scores: list[float] = []
    for label_id in range(num_labels):
        tp = confusion[label_id][label_id]
        fp = sum(confusion[row][label_id] for row in range(num_labels) if row != label_id)
        fn = sum(confusion[label_id][col] for col in range(num_labels) if col != label_id)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    return {
        "accuracy": correct / max(1, total),
        "macro_f1": sum(f1_scores) / max(1, len(f1_scores)),
        "confusion_matrix": confusion,
    }


def save_confusion_matrix_artifacts(
    *,
    confusion_matrix: list[list[int]],
    labels: list[str],
    output_dir: Path,
    name: str,
) -> dict[str, str]:
    matrix_dir = output_dir / "confusion_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    json_path = matrix_dir / f"{name}.json"
    png_path = matrix_dir / f"{name}.png"
    payload = {
        "name": name,
        "labels": labels,
        "confusion_matrix": confusion_matrix,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        image = ax.imshow(confusion_matrix, cmap="Blues")
        ax.set_title(f"Language Confusion Matrix - {name}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        max_value = max(max(row) for row in confusion_matrix) if confusion_matrix else 0
        threshold = max_value / 2 if max_value else 0
        for row_index, row in enumerate(confusion_matrix):
            for col_index, value in enumerate(row):
                color = "white" if value > threshold else "black"
                ax.text(
                    col_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color=color,
                )
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        fallback_path = matrix_dir / f"{name}.txt"
        fallback_path.write_text(
            f"無法輸出混淆矩陣圖片: {exc}\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        png_path = fallback_path

    return {
        "json": str(json_path),
        "image": str(png_path),
    }


def evaluate(
    *,
    encoder,
    classifier: WhisperLanguageClassifier,
    loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    criterion,
) -> dict[str, Any]:
    classifier.eval()
    losses: list[float] = []
    predictions: list[int] = []
    references: list[int] = []
    progress = tqdm(
        loader,
        desc="eval language head",
        disable=bool(cfg.get("disable_tqdm", False)),
    )
    with torch.no_grad():
        for batch in progress:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            with build_autocast(device, cfg):
                hidden = encoder(input_features).last_hidden_state
            logits = classifier(hidden.float())
            loss = criterion(logits, labels)
            losses.append(float(loss.detach().cpu()))
            predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
            references.extend(labels.detach().cpu().tolist())

    metrics = compute_metrics(
        predictions,
        references,
        num_labels=len(classifier.spec.labels),
    )
    metrics["loss"] = sum(losses) / max(1, len(losses))
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
    return wandb.init(
        project=os.environ["WANDB_PROJECT"],
        name=os.environ["WANDB_NAME"],
        config=cfg,
    )


def main() -> None:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    args = parse_args()
    config = load_config(args.config)
    train_cfg = get_train_config(config)
    model_cfg = train_cfg.get("model", {}) or {}
    data_cfg = train_cfg.get("data", {}) or {}
    cls_cfg = train_cfg.get("language_classifier", {}) or {}
    if not bool(cls_cfg.get("enabled", True)):
        raise ValueError("language_classifier.enabled 已關閉。")

    labels = [str(label) for label in cls_cfg.get("labels", ["zh-TW", "nan-tw"])]
    model_name_or_path = str(
        model_cfg.get("model_name_or_path") or "openai/whisper-medium"
    )
    output_dir = Path(str(cls_cfg.get("output_dir") or "artifacts/models/language_classifier"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bool(cls_cfg.get("tf32", False)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        language=str(model_cfg.get("language") or "zh"),
        task=str(model_cfg.get("task") or "transcribe"),
    )
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path).to(device)
    whisper.eval()
    for param in whisper.parameters():
        param.requires_grad = False
    encoder = whisper.model.encoder

    hidden_size = int(getattr(whisper.config, "d_model", 1024))
    spec = LanguageClassifierSpec(
        labels=tuple(labels),
        hidden_size=hidden_size,
        pooling=str(cls_cfg.get("pooling") or "mean_max"),
        hidden_ratio=float(cls_cfg.get("hidden_ratio", 0.5)),
        num_hidden_layers=int(cls_cfg.get("num_hidden_layers", 2)),
        dropout=float(cls_cfg.get("dropout", 0.1)),
    )
    classifier = WhisperLanguageClassifier(spec).to(device)

    max_train_samples = (
        args.max_train_samples
        if args.max_train_samples is not None
        else cls_cfg.get("max_train_samples")
    )
    max_eval_samples = (
        args.max_eval_samples
        if args.max_eval_samples is not None
        else cls_cfg.get("max_eval_samples")
    )
    max_test_samples = cls_cfg.get("max_test_samples")
    train_dataset = LanguageDataset(
        data_cfg=data_cfg,
        split=str(cls_cfg.get("train_split") or data_cfg.get("train_split", "train")),
        processor=processor,
        labels=labels,
        max_samples=None if max_train_samples is None else int(max_train_samples),
    )
    eval_dataset = LanguageDataset(
        data_cfg=data_cfg,
        split=str(cls_cfg.get("eval_split") or data_cfg.get("eval_split", "dev")),
        processor=processor,
        labels=labels,
        max_samples=None if max_eval_samples is None else int(max_eval_samples),
    )
    test_dataset = LanguageDataset(
        data_cfg=data_cfg,
        split=str(cls_cfg.get("test_split") or data_cfg.get("test_split", "test")),
        processor=processor,
        labels=labels,
        max_samples=None if max_test_samples is None else int(max_test_samples),
    )
    if len(train_dataset) == 0:
        raise ValueError("語言分類頭訓練資料為空，請先檢查前處理後的 language_label。")
    if len(eval_dataset) == 0:
        raise ValueError("語言分類頭驗證資料為空，請先檢查前處理後的 language_label。")
    if len(test_dataset) == 0:
        raise ValueError("語言分類頭測試資料為空，請先檢查前處理後的 language_label。")

    collator = LanguageCollator(processor)
    num_workers = int(cls_cfg.get("dataloader_num_workers", 1))
    loader_kwargs = {
        "collate_fn": collator,
        "num_workers": num_workers,
        "pin_memory": bool(cls_cfg.get("dataloader_pin_memory", device.type == "cuda")),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            cls_cfg.get("dataloader_persistent_workers", False)
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cls_cfg.get("per_device_train_batch_size", 4)),
        shuffle=True,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(cls_cfg.get("per_device_eval_batch_size", 4)),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cls_cfg.get("per_device_eval_batch_size", 4)),
        shuffle=False,
        **loader_kwargs,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=float(cls_cfg.get("learning_rate", 1.0e-4)),
        weight_decay=float(cls_cfg.get("weight_decay", 0.01)),
    )
    epochs = float(args.num_epochs if args.num_epochs is not None else cls_cfg.get("num_train_epochs", 10.0))
    update_steps = max(1, int(len(train_loader) * epochs))
    warmup_steps = int(cls_cfg.get("warmup_steps", 100))
    early_stopping_cfg = dict(cls_cfg.get("early_stopping") or {})
    early_stopping_enabled = bool(early_stopping_cfg.get("enabled", True))
    early_stopping_patience = max(1, int(early_stopping_cfg.get("patience", 3)))
    early_stopping_min_delta = float(early_stopping_cfg.get("min_delta", 0.0))
    early_stopping_metric = str(early_stopping_cfg.get("metric") or "macro_f1")
    if early_stopping_metric != "macro_f1":
        raise ValueError("分類頭早停目前只支援 macro_f1。")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        remaining = max(1, update_steps - warmup_steps)
        return max(0.0, (update_steps - step) / remaining)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    wandb_run = init_wandb(cls_cfg, output_dir)

    print(
        json.dumps(
            {
                "config": args.config,
                "model_name_or_path": model_name_or_path,
                "labels": labels,
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
                "test_samples": len(test_dataset),
                "output_dir": str(output_dir),
                "device": str(device),
                "classifier": {
                    "pooling": spec.pooling,
                    "hidden_ratio": spec.hidden_ratio,
                    "num_hidden_layers": spec.num_hidden_layers,
                    "dropout": spec.dropout,
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
    for epoch in range(1, full_epochs + 1):
        classifier.train()
        losses: list[float] = []
        progress = tqdm(
            train_loader,
            desc=f"train language head {epoch}/{full_epochs}",
            disable=bool(cls_cfg.get("disable_tqdm", False)),
        )
        for batch in progress:
            global_step += 1
            input_features = batch["input_features"].to(device)
            labels_tensor = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                with build_autocast(device, cls_cfg):
                    hidden = encoder(input_features).last_hidden_state
            logits = classifier(hidden.float())
            loss = criterion(logits, labels_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                classifier.parameters(),
                float(cls_cfg.get("max_grad_norm", 1.0)),
            )
            optimizer.step()
            scheduler.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            progress.set_postfix(loss=f"{loss_value:.4f}")
            if wandb_run is not None and global_step % int(cls_cfg.get("logging_steps", 25)) == 0:
                wandb_run.log(
                    {
                        "train/loss": loss_value,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        eval_metrics = evaluate(
            encoder=encoder,
            classifier=classifier,
            loader=eval_loader,
            device=device,
            cfg=cls_cfg,
            criterion=criterion,
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
            wandb_run.log(metrics, step=global_step)

        if improved:
            best_macro_f1 = current_macro_f1
            stale_epochs = 0
            payload = classifier.checkpoint_payload()
            payload.update(
                {
                    "model_name_or_path": model_name_or_path,
                    "best_macro_f1": best_macro_f1,
                    "epoch": epoch,
                }
            )
            torch.save(payload, output_dir / "language_classifier.pt")
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

    best_checkpoint_path = output_dir / "language_classifier.pt"
    if best_checkpoint_path.exists():
        best_payload = torch.load(best_checkpoint_path, map_location=device)
        classifier.load_state_dict(best_payload["state_dict"])
    test_metrics = evaluate(
        encoder=encoder,
        classifier=classifier,
        loader=test_loader,
        device=device,
        cfg=cls_cfg,
        criterion=criterion,
    )
    test_confusion_paths = save_confusion_matrix_artifacts(
        confusion_matrix=test_metrics["confusion_matrix"],
        labels=labels,
        output_dir=output_dir,
        name="test_final",
    )
    test_summary = {
        "split": "test",
        "best_dev_macro_f1": best_macro_f1,
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "test_confusion_matrix_paths": test_confusion_paths,
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(test_summary, ensure_ascii=False), flush=True)
    if wandb_run is not None:
        log_payload = dict(test_summary)
        image_path = test_confusion_paths.get("image")
        if image_path and str(image_path).endswith(".png"):
            import wandb

            log_payload["test/confusion_matrix"] = wandb.Image(image_path)
        wandb_run.log(log_payload, step=global_step)

    if wandb_run is not None:
        wandb_run.finish()
    print(f"已保存最佳語言分類頭: {output_dir / 'language_classifier.pt'}", flush=True)


if __name__ == "__main__":
    main()
