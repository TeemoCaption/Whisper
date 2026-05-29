from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def sanitize_adapter_name(value: str) -> str:
    text = str(value or "").strip().replace("-", "_")
    safe = "".join(char.lower() if char.isalnum() else "_" for char in text)
    return "_".join(part for part in safe.split("_") if part) or "shared"


def is_peft_enabled(peft_cfg: dict[str, Any] | None) -> bool:
    return bool((peft_cfg or {}).get("enabled", False))


def resolve_adapter_names(peft_cfg: dict[str, Any]) -> dict[str, Any]:
    scope = str(peft_cfg.get("adapter_scope") or "shared").strip().lower()
    shared_name = sanitize_adapter_name(peft_cfg.get("adapter_name") or "shared")
    language_adapters = {
        str(language): sanitize_adapter_name(name)
        for language, name in dict(peft_cfg.get("language_adapters") or {}).items()
    }

    if scope not in {"shared", "language"}:
        raise ValueError("peft.adapter_scope 只能是 shared 或 language。")

    active_language = peft_cfg.get("active_language")
    active_adapter = peft_cfg.get("active_adapter")
    if scope == "language" and active_language:
        try:
            active_name = language_adapters[str(active_language)]
        except KeyError as exc:
            raise ValueError(
                f"peft.active_language={active_language!r} 沒有對應的語言轉接模組。"
            ) from exc
    elif active_adapter:
        active_name = sanitize_adapter_name(str(active_adapter))
    else:
        active_name = shared_name

    if scope == "language":
        adapter_names = list(dict.fromkeys(language_adapters.values()))
        if active_name not in adapter_names:
            adapter_names.insert(0, active_name)
    else:
        adapter_names = [active_name]

    return {
        "adapter_scope": scope,
        "shared_adapter": shared_name,
        "language_adapters": language_adapters,
        "active_adapter": active_name,
        "adapter_names": adapter_names,
    }


def _import_peft():
    try:
        from peft import AdaLoraConfig, LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "需要安裝 peft 才能啟用低秩適應；請先執行 pip install -r requirements.txt。"
        ) from exc
    return AdaLoraConfig, LoraConfig, get_peft_model


def build_peft_config(peft_cfg: dict[str, Any]):
    AdaLoraConfig, LoraConfig, _ = _import_peft()
    method = str(peft_cfg.get("method") or "lora").strip().lower()
    target_modules = list(peft_cfg.get("target_modules") or DEFAULT_TARGET_MODULES)
    common_kwargs: dict[str, Any] = {
        "target_modules": target_modules,
        "inference_mode": False,
        "bias": str(peft_cfg.get("bias") or "none"),
        "modules_to_save": list(peft_cfg.get("modules_to_save") or []) or None,
    }

    if method == "lora":
        lora_cfg = dict(peft_cfg.get("lora") or {})
        return LoraConfig(
            r=int(lora_cfg.get("r", 32)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 64)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            **common_kwargs,
        )
    if method == "adalora":
        adalora_cfg = dict(peft_cfg.get("adalora") or {})
        kwargs = {
            "init_r": int(adalora_cfg.get("init_r", 32)),
            "target_r": int(adalora_cfg.get("target_r", 16)),
            "beta1": float(adalora_cfg.get("beta1", 0.85)),
            "beta2": float(adalora_cfg.get("beta2", 0.85)),
            "tinit": int(adalora_cfg.get("tinit", 200)),
            "tfinal": int(adalora_cfg.get("tfinal", 1000)),
            "deltaT": int(adalora_cfg.get("deltaT", 10)),
            "lora_alpha": int(adalora_cfg.get("lora_alpha", 64)),
            "lora_dropout": float(adalora_cfg.get("lora_dropout", 0.05)),
            **common_kwargs,
        }
        if adalora_cfg.get("total_step") is not None:
            kwargs["total_step"] = int(adalora_cfg["total_step"])
        return AdaLoraConfig(**kwargs)
    raise ValueError("peft.method 只能是 lora 或 adalora。")


def count_parameters(model) -> dict[str, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return {
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
    }


def apply_peft_adapters(model, peft_cfg: dict[str, Any]):
    _, _, get_peft_model = _import_peft()
    adapter_info = resolve_adapter_names(peft_cfg)
    peft_config = build_peft_config(peft_cfg)
    active_adapter = adapter_info["active_adapter"]

    model = get_peft_model(model, peft_config, adapter_name=active_adapter)
    for adapter_name in adapter_info["adapter_names"]:
        if adapter_name == active_adapter:
            continue
        model.add_adapter(adapter_name, peft_config)
    model.set_adapter(active_adapter)

    method = str(peft_cfg.get("method") or "lora").strip().lower()
    adapter_info.update(
        {
            "enabled": True,
            "method": method,
            "target_modules": list(peft_cfg.get("target_modules") or DEFAULT_TARGET_MODULES),
            "save_all_adapters": bool(peft_cfg.get("save_all_adapters", False)),
            **count_parameters(model),
        }
    )
    return model, adapter_info


def update_adalora_rank_allocation(model, global_step: int) -> bool:
    if global_step <= 0:
        return False
    candidates = [model, getattr(model, "base_model", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        method = getattr(candidate, "update_and_allocate", None)
        if callable(method):
            trainable_params = [
                param for param in candidate.parameters() if param.requires_grad
            ]
            if trainable_params and not any(
                param.grad is not None for param in trainable_params
            ):
                return False
            for param in trainable_params:
                if param.grad is None:
                    param.grad = param.new_zeros(param.shape)
            method(global_step)
            return True
    return False


def save_peft_artifacts(model, final_dir: Path, peft_info: dict[str, Any]) -> None:
    final_dir = Path(final_dir)
    adapter_root = final_dir / "adapters"
    adapter_root.mkdir(parents=True, exist_ok=True)

    selected_adapters = list(peft_info.get("adapter_names") or [])
    if not bool(peft_info.get("save_all_adapters", False)):
        selected_adapters = [str(peft_info.get("active_adapter") or "shared")]

    saved_adapters: list[str] = []
    for adapter_name in selected_adapters:
        adapter_dir = adapter_root / sanitize_adapter_name(adapter_name)
        model.save_pretrained(
            str(adapter_dir),
            selected_adapters=[adapter_name],
            safe_serialization=True,
        )
        saved_adapters.append(str(adapter_dir))

    manifest = {
        key: value
        for key, value in peft_info.items()
        if key not in {"trainable_parameters", "total_parameters"}
    }
    manifest.update(
        {
            **count_parameters(model),
            "adapter_root": str(adapter_root),
            "saved_adapters": saved_adapters,
        }
    )
    (final_dir / "adapter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _decision_value(decision: Any, key: str, default: Any = None) -> Any:
    if isinstance(decision, dict):
        return decision.get(key, default)
    return getattr(decision, key, default)


def activate_routed_adapters(
    model,
    decision: Any,
    *,
    mixed_adapter_name: str = "confidence_mixed",
    combination_type: str = "linear",
) -> dict[str, Any]:
    adapter_names = [
        sanitize_adapter_name(name)
        for name in _decision_value(decision, "adapter_names", [])
    ]
    weights = [float(value) for value in _decision_value(decision, "weights", [])]
    routing_mode = str(_decision_value(decision, "routing_mode", "single"))

    if not adapter_names:
        raise ValueError("路由結果缺少 adapter_names。")
    if not weights:
        weights = [1.0 / len(adapter_names)] * len(adapter_names)
    if len(adapter_names) != len(weights):
        raise ValueError("路由結果的 adapter_names 與 weights 長度不一致。")

    if len(adapter_names) == 1 or routing_mode in {"single", "shared"}:
        model.set_adapter(adapter_names[0])
        return {
            "active_adapter": adapter_names[0],
            "adapter_names": adapter_names,
            "weights": weights,
            "routing_mode": routing_mode,
            "mixed": False,
        }

    if not hasattr(model, "add_weighted_adapter"):
        raise RuntimeError("目前模型不支援加權混合 adapter；請改用單一或共享路由。")

    mixed_name = sanitize_adapter_name(mixed_adapter_name)
    if hasattr(model, "delete_adapter"):
        try:
            model.delete_adapter(mixed_name)
        except Exception:
            pass
    model.add_weighted_adapter(
        adapter_names,
        weights,
        mixed_name,
        combination_type=combination_type,
    )
    model.set_adapter(mixed_name)
    return {
        "active_adapter": mixed_name,
        "adapter_names": adapter_names,
        "weights": weights,
        "routing_mode": routing_mode,
        "mixed": True,
    }
