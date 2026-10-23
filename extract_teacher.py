"""
Offline teacher-knowledge extraction for hierarchical multi-label distillation.

Examples
--------
Single GPU, extract all knowledge:

CUDA_VISIBLE_DEVICES=2 python extract_teacher.py \
  --dataset wos \
  --knowledge_types attention

Multi-GPU, resumable extraction:

torchrun --nproc_per_node=4 extract_teacher.py \
  --dataset wos \
  --knowledge_types logits,features,attention

The extractor never changes the source JSONL. Shards use fixed global
dataset-index ranges, so their logical order is independent of worker count or
completion order.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from train_baseline import (
    DATASET_ZOO,
    LEVEL_TOKENS,
    HierarchicalDataset,
    HierarchicalTeacher,
    add_level_special_tokens,
    assert_preprocess_meta_compatible,
    build_level_offsets,
    load_preprocess_meta,
    normalize_layer_label2id,
    seed_everything,
)


DEFAULT_TEACHER_CHECKPOINTS = {
    "wos": "/data1/lwq/agt_plus/output_plus/"
           "wos_qwen8b_local0_global1_collator-epoch7/best_global_macro",
    "rcv1": "/data1/lwq/agt_plus/output_plus/"
            "rcv1_qwen8b_local0.5_global1/best_global_macro",
    "nyt": "/data1/lwq/agt_plus/output_plus/"
          "nyt_qwen8b_local0.5_global1/best_global_macro",
    "bgc": "/data1/lwq/agt_plus/output_plus/"
          "bgc_qwen8b_local0_global1_collator/best_global_macro",

}

KNOWLEDGE_TYPES = {"logits", "features", "attention"}
SCHEMA_VERSION = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract sharded, resumable knowledge from a fine-tuned hierarchical teacher."
    )
    parser.add_argument("--dataset", default="wos", choices=sorted(DATASET_ZOO))
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help="Teacher best-model directory. Overrides DEFAULT_TEACHER_CHECKPOINTS.",
    )
    parser.add_argument("--data_dir", default=None, help="Override DATASET_ZOO data path.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Knowledge root. Defaults to <checkpoint_dir>/teacher_knowledge_train.",
    )
    parser.add_argument(
        "--knowledge_types",
        default="logits,features,attention",
        help="Comma-separated subset of logits,features,attention.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=2048)
    parser.add_argument("--attention_topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite_shards", action="store_true")
    parser.add_argument(
        "--allow_full_attention_fallback",
        action="store_true",
        help="If selected-layer replay is unsupported, allow a memory-hungry full-model attention pass.",
    )
    return parser.parse_args()


def parse_knowledge_types(raw: str) -> list[str]:
    values = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("--knowledge_types cannot be empty")
    unknown = sorted(set(values) - KNOWLEDGE_TYPES)
    if unknown:
        raise ValueError(f"Unknown knowledge types: {unknown}")
    return list(dict.fromkeys(values))


def trim_right_padding_collator(
    features: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    batch = {
        key: torch.stack([feature[key] for feature in features])
        for key in features[0]
    }
    max_valid_length = int(batch["attention_mask"].sum(dim=1).max().item())
    if max_valid_length <= 0:
        raise RuntimeError("A batch contains no valid input tokens.")
    batch["input_ids"] = batch["input_ids"][:, :max_valid_length].contiguous()
    batch["attention_mask"] = batch["attention_mask"][
        :, :max_valid_length
    ].contiguous()
    return batch


def distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("Teacher extraction requires CUDA.")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return
    finite = torch.isfinite(tensor)
    if bool(finite.all().item()):
        return
    invalid_count = int((~finite).sum().item())
    raise RuntimeError(
        f"{name} contains {invalid_count} NaN/Inf value(s); refusing to save or use it."
    )


def find_teacher_config_path(checkpoint_dir: str) -> str:
    candidates = [
        os.path.join(checkpoint_dir, "teacher_config.json"),
        os.path.join(os.path.dirname(checkpoint_dir), "teacher_config.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Could not find teacher_config.json in the checkpoint or its parent."
    )


def teacher_checkpoint_fingerprint(
    checkpoint_dir: str,
) -> dict[str, Any]:
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    teacher_config_path = find_teacher_config_path(checkpoint_dir)
    with open(teacher_config_path, "r", encoding="utf-8") as handle:
        teacher_config = json.load(handle)
    model_path_value = teacher_config.get("model_path")
    if not isinstance(model_path_value, str) or not model_path_value:
        raise ValueError("teacher_config.json must contain a non-empty model_path.")
    expanded_model_path = os.path.expanduser(model_path_value)
    if not os.path.isabs(expanded_model_path):
        expanded_model_path = os.path.join(
            os.path.dirname(teacher_config_path),
            expanded_model_path,
        )
    model_path = os.path.abspath(expanded_model_path)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Teacher base-model directory does not exist: {model_path}"
        )

    adapter_candidates = [
        os.path.join(checkpoint_dir, "adapter_model.safetensors"),
        os.path.join(checkpoint_dir, "adapter_model.bin"),
    ]
    adapter_path = next(
        (path for path in adapter_candidates if os.path.exists(path)),
        None,
    )
    required_paths = {
        "checkpoint/adapter_model": adapter_path,
        "checkpoint/adapter_config.json": os.path.join(
            checkpoint_dir,
            "adapter_config.json",
        ),
        "checkpoint/classifier_head.pt": os.path.join(
            checkpoint_dir,
            "classifier_head.pt",
        ),
        "checkpoint/teacher_config.json": teacher_config_path,
        "base_model/config.json": os.path.join(model_path, "config.json"),
    }

    index_path = None
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        candidate = os.path.join(model_path, index_name)
        if os.path.exists(candidate):
            index_path = candidate
            required_paths[f"base_model/{index_name}"] = candidate
            with open(candidate, "r", encoding="utf-8") as handle:
                index_data = json.load(handle)
            weight_map = index_data.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(
                    f"Base-model weight index has no valid weight_map: {candidate}"
                )
            for shard_name in sorted(set(weight_map.values())):
                if not isinstance(shard_name, str) or not shard_name:
                    raise ValueError(
                        f"Invalid shard name in base-model index: {shard_name!r}"
                    )
                shard_path = os.path.abspath(os.path.join(model_path, shard_name))
                if os.path.commonpath([model_path, shard_path]) != model_path:
                    raise ValueError(
                        f"Base-model index references a path outside model_path: "
                        f"{shard_name}"
                    )
                required_paths[f"base_model/{shard_name}"] = shard_path
            break
    if index_path is None:
        for weight_name in ("model.safetensors", "pytorch_model.bin"):
            candidate = os.path.join(model_path, weight_name)
            if os.path.exists(candidate):
                required_paths[f"base_model/{weight_name}"] = candidate
                break
        else:
            raise FileNotFoundError(
                "Could not find base-model weights or a supported weight index in "
                f"{model_path}."
            )

    missing = [
        name
        for name, path in required_paths.items()
        if path is None or not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError(
            f"Teacher checkpoint lacks required fingerprint files: {missing}"
        )

    optional_names = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]
    selected_paths: dict[str, str] = {
        name: str(path)
        for name, path in required_paths.items()
        if path is not None
    }
    for name in optional_names:
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            selected_paths[f"checkpoint/{name}"] = path

    files: dict[str, dict[str, Any]] = {}
    for name, path in sorted(selected_paths.items()):
        files[name] = {
            "path": os.path.abspath(path),
            "size_bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
        }
    aggregate_payload = json.dumps(
        {
            name: {
                "size_bytes": entry["size_bytes"],
                "sha256": entry["sha256"],
            }
            for name, entry in files.items()
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "sha256": sha256_bytes(aggregate_payload),
        "files": files,
        "base_model_path": model_path,
    }


def is_attention_implementation_compat_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "attn_implementation",
            "attention implementation",
            "attention_implementation",
        )
    )


def uid_hex_to_tensor(values: list[str]) -> torch.Tensor:
    return torch.tensor([list(bytes.fromhex(value)) for value in values], dtype=torch.uint8)


def tensor_uid_hex(values: torch.Tensor) -> list[str]:
    return [bytes(row.tolist()).hex() for row in values.cpu()]


def atomic_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp-{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def atomic_safetensors(
    path: str,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp-{os.getpid()}.safetensors"
    contiguous = {
        key: value.detach().cpu().contiguous()
        for key, value in tensors.items()
    }
    save_file(contiguous, temp_path, metadata=metadata)
    os.replace(temp_path, path)


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json_if_exists(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    return read_json(path)


def find_teacher_config(checkpoint_dir: str) -> dict[str, Any]:
    return read_json(find_teacher_config_path(checkpoint_dir))


def read_head_training_status(
    checkpoint_dir: str,
    teacher_config: dict[str, Any],
) -> dict[str, bool]:
    classifier_path = os.path.join(checkpoint_dir, "classifier_head.pt")
    classifier_state = torch.load(classifier_path, map_location="cpu")
    status = {
        "has_trained_local_heads": bool(
            classifier_state.get(
                "has_trained_local_heads",
                teacher_config.get(
                    "has_trained_local_heads",
                    float(teacher_config.get("local_loss_weight", 0.0)) > 0,
                ),
            )
        ),
        "has_trained_global_head": bool(
            classifier_state.get(
                "has_trained_global_head",
                teacher_config.get(
                    "has_trained_global_head",
                    float(teacher_config.get("global_loss_weight", 0.0)) > 0,
                ),
            )
        ),
    }
    for key, actual in status.items():
        if key in teacher_config and bool(teacher_config[key]) != actual:
            raise RuntimeError(
                f"{key} differs between teacher_config.json and classifier_head.pt."
            )
    return status


def canonical_sample_payload(
    dataset: str,
    split: str,
    source_line_no: int,
    item: dict[str, Any],
    num_levels: int,
) -> bytes:
    payload = {
        "dataset": dataset,
        "split": split,
        "source_line_no": int(source_line_no),
        "query": item["query"],
        "label": item["label"],
        "label_mask": item.get("label_mask", [1] * num_levels),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_sample_identities(
    data_path: str,
    dataset: str,
    split: str,
    num_levels: int,
) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as handle:
        for source_line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            uid = sha256_bytes(
                canonical_sample_payload(
                    dataset,
                    split,
                    source_line_no,
                    item,
                    num_levels,
                )
            )
            identities.append(
                {
                    "dataset_idx": len(identities),
                    "source_line_no": source_line_no,
                    "sample_uid": uid,
                    "query_hash": sha256_bytes(item["query"].encode("utf-8")),
                }
            )
    return identities


def tokenizer_fingerprint(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_json = None
    if backend is not None and hasattr(backend, "to_str"):
        backend_json = backend.to_str()
    fingerprint_payload = {
        "class": type(tokenizer).__name__,
        "vocab": sorted(
            (str(token), int(token_id))
            for token, token_id in vocab.items()
        ),
        "backend_tokenizer": backend_json,
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    payload = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def find_decoder_layers(backbone: nn.Module) -> nn.ModuleList:
    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, module in backbone.named_modules():
        layers = getattr(module, "layers", None)
        if (
            isinstance(layers, nn.ModuleList)
            and len(layers) > 0
            and hasattr(layers[-1], "self_attn")
        ):
            candidates.append((name, layers))
    if not candidates:
        raise RuntimeError(
            "Could not locate decoder layers containing self_attn modules."
        )
    candidates.sort(key=lambda item: (len(item[1]), len(item[0])), reverse=True)
    return candidates[0][1]


def configured_feature_layer(teacher_config: dict[str, Any]) -> int:
    if not bool(teacher_config.get("use_hidden_states", False)):
        return -1
    return int(teacher_config.get("hidden_layer_idx", -1))


class LayerAttentionCapture:
    """Capture the normalized input actually received by a selected self-attention layer."""

    def __init__(self, backbone: nn.Module, layer_idx: int = -1):
        layers = find_decoder_layers(backbone)
        resolved_layer_idx = layer_idx if layer_idx >= 0 else len(layers) + layer_idx
        if not 0 <= resolved_layer_idx < len(layers):
            raise ValueError(
                f"Attention layer index {layer_idx} resolves to {resolved_layer_idx}, "
                f"outside [0, {len(layers)})."
            )
        self.layer_idx = int(layer_idx)
        self.resolved_layer_idx = int(resolved_layer_idx)
        self.num_decoder_layers = len(layers)
        self.layer = layers[resolved_layer_idx]
        self.self_attn = self.layer.self_attn
        self.args: tuple[Any, ...] | None = None
        self.kwargs: dict[str, Any] | None = None
        self.hidden_states: torch.Tensor | None = None
        self._is_replaying = False
        try:
            self.hook = self.self_attn.register_forward_pre_hook(
                self._capture_with_kwargs,
                with_kwargs=True,
            )
            self.with_kwargs = True
        except TypeError:
            self.hook = self.self_attn.register_forward_pre_hook(
                self._capture_args_only
            )
            self.with_kwargs = False

    def clear(self) -> None:
        self.args = None
        self.kwargs = None
        self.hidden_states = None

    def close(self) -> None:
        self.hook.remove()

    def _capture_with_kwargs(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if self._is_replaying:
            return
        self.args = args
        try:
            bound = inspect.signature(self.self_attn.forward).bind_partial(
                *args,
                **kwargs,
            )
            captured = dict(bound.arguments)
        except (TypeError, ValueError):
            captured = dict(kwargs)
        hidden_states = captured.pop("hidden_states", None)
        nested_kwargs = captured.pop("kwargs", None)
        if isinstance(nested_kwargs, dict):
            captured.update(nested_kwargs)
        self.kwargs = captured
        if torch.is_tensor(hidden_states):
            self.hidden_states = hidden_states
        elif args and torch.is_tensor(args[0]):
            self.hidden_states = args[0]
        elif torch.is_tensor(kwargs.get("hidden_states")):
            self.hidden_states = kwargs["hidden_states"]

    def _capture_args_only(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
    ) -> None:
        if self._is_replaying:
            return
        self.args = args
        try:
            bound = inspect.signature(self.self_attn.forward).bind_partial(*args)
            captured = dict(bound.arguments)
        except (TypeError, ValueError):
            captured = {}
        hidden_states = captured.pop("hidden_states", None)
        nested_kwargs = captured.pop("kwargs", None)
        if isinstance(nested_kwargs, dict):
            captured.update(nested_kwargs)
        self.kwargs = captured
        if torch.is_tensor(hidden_states):
            self.hidden_states = hidden_states
        elif args and torch.is_tensor(args[0]):
            self.hidden_states = args[0]

    @staticmethod
    def _call_with_supported_kwargs(
        module: nn.Module,
        hidden_states: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> Any:
        signature = inspect.signature(module.forward)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        call_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key == "hidden_states":
                continue
            if accepts_var_kwargs or key in signature.parameters:
                call_kwargs[key] = value
        if accepts_var_kwargs or "output_attentions" in signature.parameters:
            call_kwargs["output_attentions"] = True
        return module(hidden_states=hidden_states, **call_kwargs)

    @staticmethod
    def _find_attention_tensor(result: Any, seq_len: int) -> torch.Tensor:
        candidates: list[torch.Tensor] = []
        if torch.is_tensor(result):
            candidates.append(result)
        elif isinstance(result, (tuple, list)):
            candidates.extend(value for value in result if torch.is_tensor(value))
        else:
            for name in ("attn_weights", "attention_weights", "attentions"):
                value = getattr(result, name, None)
                if torch.is_tensor(value):
                    candidates.append(value)
        for value in candidates:
            if (
                value.ndim == 4
                and value.shape[-2] == seq_len
                and value.shape[-1] == seq_len
            ):
                return value
        shapes = [tuple(value.shape) for value in candidates]
        raise RuntimeError(
            "Selected-layer attention replay did not return a [batch,heads,seq,seq] "
            f"tensor. Tensor outputs were: {shapes}"
        )

    def replay(self) -> torch.Tensor:
        if self.args is None:
            raise RuntimeError(
                "Selected decoder layer was not captured during the model forward."
            )
        if self.hidden_states is None:
            raise RuntimeError(
                "Could not identify selected-layer hidden_states in captured args."
            )
        hidden_states = self.hidden_states
        kwargs = dict(self.kwargs or {})
        self._is_replaying = True
        try:
            result = self._call_with_supported_kwargs(
                self.self_attn,
                hidden_states,
                kwargs,
            )
        finally:
            self._is_replaying = False
        return self._find_attention_tensor(result, hidden_states.shape[1])


def validate_attention_replay(
    backbone: nn.Module,
    capture: LayerAttentionCapture,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_levels: int,
    max_query_tokens: int = 64,
    atol: float = 5e-4,
    rtol: float = 5e-3,
) -> dict[str, float]:
    """Compare replay against a real eager forward on one short sequence."""
    if input_ids.ndim != 2 or input_ids.shape[0] < 1:
        raise ValueError("Attention replay validation needs a non-empty batch.")
    if max_query_tokens <= 0:
        raise ValueError("max_query_tokens must be positive.")

    valid_len = int(attention_mask[0].sum().item())
    query_len = valid_len - num_levels
    if query_len <= 0:
        raise RuntimeError("Validation sample has no query tokens.")
    kept_query_len = min(query_len, max_query_tokens)
    short_ids = torch.cat(
        [
            input_ids[0, :kept_query_len],
            input_ids[0, query_len:valid_len],
        ],
        dim=0,
    ).unsqueeze(0)
    short_mask = torch.ones_like(short_ids)

    was_training = backbone.training
    backbone.eval()
    capture.clear()
    try:
        with torch.inference_mode():
            reference_outputs = backbone(
                input_ids=short_ids,
                attention_mask=short_mask,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
            reference_attentions = getattr(
                reference_outputs,
                "attentions",
                None,
            )
            if not reference_attentions:
                raise RuntimeError(
                    "Attention validation forward returned no attentions. "
                    "Ensure the model uses eager attention."
                )
            reference_value = reference_attentions[
                capture.resolved_layer_idx
            ]
            if not torch.is_tensor(reference_value):
                raise RuntimeError(
                    "Attention validation forward returned a non-tensor selected-layer "
                    "attention."
                )
            reference = reference_value.float()
            replayed = capture.replay().float()
    finally:
        backbone.train(was_training)

    if reference.shape != replayed.shape:
        raise RuntimeError(
            "Attention replay validation shape mismatch: "
            f"reference={tuple(reference.shape)}, replay={tuple(replayed.shape)}"
        )
    assert_finite_tensor("reference attention", reference)
    assert_finite_tensor("replayed attention", replayed)
    difference = (reference - replayed).abs()
    max_abs_error = float(difference.max().item())
    mean_abs_error = float(difference.mean().item())
    if not torch.allclose(reference, replayed, atol=atol, rtol=rtol):
        raise RuntimeError(
            "Attention replay is numerically inconsistent with the real selected-layer "
            "attention: "
            f"max_abs_error={max_abs_error:.8f}, "
            f"mean_abs_error={mean_abs_error:.8f}, atol={atol}, rtol={rtol}."
        )
    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "validation_seq_len": float(short_ids.shape[1]),
    }


def extract_model_outputs(
    model: HierarchicalTeacher,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    post_ln_features = torch.stack(
        [model.post_ln(feature) for feature in outputs["level_features"]],
        dim=1,
    )
    return outputs, post_ln_features


def topk_attention_knowledge(
    attentions: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    level_token_ids: list[int],
    topk: int,
    offsets_by_sample: list[list[tuple[int, int]]] | None,
) -> dict[str, torch.Tensor]:
    if attentions.ndim != 4:
        raise ValueError(f"Expected 4D attention tensor, got {attentions.shape}")
    batch_size, _, seq_len, _ = attentions.shape
    num_levels = len(level_token_ids)
    mean_attention = attentions.float().mean(dim=1)

    positions = torch.full(
        (batch_size, num_levels, topk),
        -1,
        dtype=torch.int32,
        device=input_ids.device,
    )
    token_ids = torch.full_like(positions, -1)
    valid_mask = torch.zeros_like(positions, dtype=torch.bool)
    weights = torch.zeros(
        (batch_size, num_levels, topk),
        dtype=torch.float32,
        device=input_ids.device,
    )
    other = torch.ones(
        (batch_size, num_levels),
        dtype=torch.float32,
        device=input_ids.device,
    )
    text_mass = torch.zeros_like(other)
    char_offsets = torch.full(
        (batch_size, num_levels, topk, 2),
        -1,
        dtype=torch.int32,
        device=input_ids.device,
    )

    for batch_idx in range(batch_size):
        valid_len = int(attention_mask[batch_idx].sum().item())
        query_len = valid_len - num_levels
        if query_len <= 0:
            raise RuntimeError(f"Sample {batch_idx} has no query tokens.")
        level_positions = []
        for token_id in level_token_ids:
            matches = (input_ids[batch_idx, :valid_len] == token_id).nonzero(
                as_tuple=True
            )[0]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one level token id {token_id}, got {len(matches)}."
                )
            level_positions.append(int(matches[0].item()))

        for level_idx, query_position in enumerate(level_positions):
            text_scores = mean_attention[
                batch_idx,
                query_position,
                :query_len,
            ].clamp_min(0)
            mass = text_scores.sum()
            if not torch.isfinite(mass) or mass.item() <= 0:
                raise RuntimeError(
                    f"Invalid text attention mass at sample={batch_idx}, level={level_idx}."
                )
            normalized = text_scores / mass
            actual_k = min(topk, query_len)
            top_values, top_positions = torch.topk(
                normalized,
                k=actual_k,
                largest=True,
                sorted=True,
            )
            positions[batch_idx, level_idx, :actual_k] = top_positions.to(torch.int32)
            token_ids[batch_idx, level_idx, :actual_k] = input_ids[
                batch_idx, top_positions
            ].to(torch.int32)
            valid_mask[batch_idx, level_idx, :actual_k] = True
            weights[batch_idx, level_idx, :actual_k] = top_values
            other[batch_idx, level_idx] = (1.0 - top_values.sum()).clamp_min(0)
            text_mass[batch_idx, level_idx] = mass

            if offsets_by_sample is not None:
                sample_offsets = offsets_by_sample[batch_idx]
                for slot, token_position in enumerate(top_positions.tolist()):
                    if token_position < len(sample_offsets):
                        start, end = sample_offsets[token_position]
                        char_offsets[batch_idx, level_idx, slot, 0] = int(start)
                        char_offsets[batch_idx, level_idx, slot, 1] = int(end)

    return {
        "topk_positions": positions,
        "topk_token_ids": token_ids,
        "topk_valid_mask": valid_mask,
        "teacher_topk_attention_mean": weights.to(torch.float16),
        "teacher_other_attention_mean": other.to(torch.float16),
        "teacher_text_attention_mass": text_mass.to(torch.float16),
        "topk_char_offsets": char_offsets,
    }


def encode_offsets(
    tokenizer: Any,
    queries: list[str],
    expected_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_levels: int,
    truncate_mode: str,
) -> list[list[tuple[int, int]]] | None:
    results: list[list[tuple[int, int]]] = []
    for batch_idx, query in enumerate(queries):
        try:
            encoded = tokenizer(
                query,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except (NotImplementedError, TypeError, ValueError):
            return None
        ids = list(encoded["input_ids"])
        offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
        query_len = int(attention_mask[batch_idx].sum().item()) - num_levels
        if len(ids) > query_len:
            if truncate_mode == "head":
                ids = ids[:query_len]
                offsets = offsets[:query_len]
            else:
                head_len = query_len // 2
                tail_len = query_len - head_len
                ids = ids[:head_len] + ids[-tail_len:]
                offsets = offsets[:head_len] + offsets[-tail_len:]
        expected = expected_input_ids[batch_idx, :query_len].detach().cpu().tolist()
        if ids != expected:
            raise RuntimeError(
                "Offset-mapping tokenization differs from model tokenization at "
                f"batch sample {batch_idx}."
            )
        results.append(offsets)
    return results


def expected_shard_schema(
    module_name: str,
    num_samples: int,
    manifest: dict[str, Any],
) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    num_levels = int(manifest["num_levels"])
    total_classes = int(manifest["total_classes"])
    topk = int(manifest["attention_topk"])
    schema: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {
        "dataset_idx": (torch.int64, (num_samples,)),
        "sample_uid": (torch.uint8, (num_samples, 32)),
    }
    if module_name == "sample_index":
        schema.update(
            {
                "source_line_no": (torch.int64, (num_samples,)),
                "query_hash": (torch.uint8, (num_samples, 32)),
            }
        )
    elif module_name == "logits":
        schema["teacher_global_logits"] = (
            torch.float16,
            (num_samples, total_classes),
        )
        if bool(manifest["has_trained_local_heads"]):
            schema["teacher_local_logits"] = (
                torch.float16,
                (num_samples, total_classes),
            )
    elif module_name == "features":
        hidden_size = manifest.get("teacher_hidden_size")
        if hidden_size is None:
            raise ValueError("teacher_hidden_size is unknown during shard validation.")
        schema["teacher_level_features"] = (
            torch.float16,
            (num_samples, num_levels, int(hidden_size)),
        )
    elif module_name == "attention":
        schema.update(
            {
                "topk_positions": (
                    torch.int32,
                    (num_samples, num_levels, topk),
                ),
                "topk_token_ids": (
                    torch.int32,
                    (num_samples, num_levels, topk),
                ),
                "topk_valid_mask": (
                    torch.bool,
                    (num_samples, num_levels, topk),
                ),
                "teacher_topk_attention_mean": (
                    torch.float16,
                    (num_samples, num_levels, topk),
                ),
                "teacher_other_attention_mean": (
                    torch.float16,
                    (num_samples, num_levels),
                ),
                "teacher_text_attention_mass": (
                    torch.float16,
                    (num_samples, num_levels),
                ),
                "topk_char_offsets": (
                    torch.int32,
                    (num_samples, num_levels, topk, 2),
                ),
            }
        )
    else:
        raise ValueError(f"Unknown shard module: {module_name}")
    return schema


def validate_existing_shard(
    tensor_path: str,
    sidecar_path: str,
    expected_start: int,
    expected_end: int,
    module_name: str,
    manifest: dict[str, Any],
    expected_sample_uids: torch.Tensor,
    checkpoint_fingerprint: str,
) -> bool:
    if not os.path.exists(tensor_path) or not os.path.exists(sidecar_path):
        return False
    try:
        sidecar = read_json(sidecar_path)
        checks = (
            int(sidecar.get("schema_version", -1)) == SCHEMA_VERSION,
            sidecar.get("module") == module_name,
            int(sidecar.get("start_idx", -1)) == expected_start,
            int(sidecar.get("end_idx", -1)) == expected_end,
            sidecar.get("teacher_checkpoint_fingerprint")
            == checkpoint_fingerprint,
            sidecar.get("sha256") == sha256_file(tensor_path),
        )
        if not all(checks):
            return False
        expected_count = expected_end - expected_start
        schema = expected_shard_schema(
            module_name,
            expected_count,
            manifest,
        )
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            if actual_keys != set(schema):
                return False
            tensors: dict[str, torch.Tensor] = {}
            for key, (expected_dtype, expected_shape) in schema.items():
                value = handle.get_tensor(key)
                if value.dtype != expected_dtype:
                    return False
                if tuple(value.shape) != expected_shape:
                    return False
                if value.is_floating_point() or value.is_complex():
                    if not bool(torch.isfinite(value).all().item()):
                        return False
                tensors[key] = value
            expected_indices = torch.arange(
                expected_start,
                expected_end,
                dtype=torch.int64,
            )
            if not torch.equal(tensors["dataset_idx"], expected_indices):
                return False
            if not torch.equal(
                tensors["sample_uid"],
                expected_sample_uids,
            ):
                return False
        return True
    except Exception:
        return False


def shard_paths(output_dir: str, module_name: str, shard_id: int) -> tuple[str, str]:
    stem = f"shard-{shard_id:05d}"
    module_dir = os.path.join(output_dir, module_name)
    return (
        os.path.join(module_dir, f"{stem}.safetensors"),
        os.path.join(module_dir, f"{stem}.json"),
    )


def save_module_shard(
    output_dir: str,
    module_name: str,
    shard_id: int,
    start_idx: int,
    end_idx: int,
    common: dict[str, torch.Tensor],
    payload: dict[str, torch.Tensor],
    checkpoint_fingerprint: str,
) -> None:
    tensor_path, sidecar_path = shard_paths(output_dir, module_name, shard_id)
    tensors = {**common, **payload}
    for key, value in tensors.items():
        assert_finite_tensor(f"{module_name}/{key}", value)
    metadata = {
        "schema_version": str(SCHEMA_VERSION),
        "module": module_name,
        "shard_id": str(shard_id),
        "start_idx": str(start_idx),
        "end_idx": str(end_idx),
        "teacher_checkpoint_fingerprint": checkpoint_fingerprint,
    }
    atomic_safetensors(tensor_path, tensors, metadata)
    atomic_json(
        sidecar_path,
        {
            **metadata,
            "shard_id": shard_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "num_samples": end_idx - start_idx,
            "teacher_checkpoint_fingerprint": checkpoint_fingerprint,
            "sha256": sha256_file(tensor_path),
        },
    )


def load_teacher(
    checkpoint_dir: str,
    teacher_config: dict[str, Any],
    tokenizer: Any,
    need_attention: bool,
    device: torch.device,
) -> HierarchicalTeacher:
    model_path = teacher_config["model_path"]
    can_use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if can_use_bf16 else torch.float16
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if need_attention:
        load_kwargs["attn_implementation"] = "eager"
    try:
        base_model = AutoModel.from_pretrained(model_path, **load_kwargs)
    except (TypeError, ValueError) as exc:
        if not is_attention_implementation_compat_error(exc):
            raise
        load_kwargs.pop("attn_implementation", None)
        base_model = AutoModel.from_pretrained(model_path, **load_kwargs)
        if need_attention and hasattr(base_model, "set_attn_implementation"):
            base_model.set_attn_implementation("eager")
    base_model.config.use_cache = False
    current_embeddings = base_model.get_input_embeddings().num_embeddings
    if len(tokenizer) > current_embeddings:
        base_model.resize_token_embeddings(len(tokenizer))
    backbone = PeftModel.from_pretrained(base_model, checkpoint_dir)

    level_classes = [int(value) for value in teacher_config["level_classes"]]
    level_offsets = [int(value) for value in teacher_config["level_offsets"]]
    model = HierarchicalTeacher(
        backbone=backbone,
        tokenizer=tokenizer,
        num_levels=int(teacher_config["num_levels"]),
        level_classes=level_classes,
        level_offsets=level_offsets,
        hidden_layer_idx=int(teacher_config.get("hidden_layer_idx", -1)),
        use_hidden_states=bool(teacher_config.get("use_hidden_states", False)),
        dropout=0.0,
    )
    classifier_path = os.path.join(checkpoint_dir, "classifier_head.pt")
    classifier_state = torch.load(classifier_path, map_location="cpu")
    has_trained_local_heads = bool(
        classifier_state.get(
            "has_trained_local_heads",
            teacher_config.get(
                "has_trained_local_heads",
                float(teacher_config.get("local_loss_weight", 0.0)) > 0,
            ),
        )
    )
    has_trained_global_head = bool(
        classifier_state.get(
            "has_trained_global_head",
            teacher_config.get(
                "has_trained_global_head",
                float(teacher_config.get("global_loss_weight", 0.0)) > 0,
            ),
        )
    )
    for key, actual in (
        ("has_trained_local_heads", has_trained_local_heads),
        ("has_trained_global_head", has_trained_global_head),
    ):
        if key in teacher_config and bool(teacher_config[key]) != actual:
            raise RuntimeError(
                f"{key} differs between teacher_config.json and classifier_head.pt."
            )
    if list(classifier_state["level_classes"]) != level_classes:
        raise RuntimeError("classifier_head.pt level_classes mismatch.")
    if list(classifier_state["level_offsets"]) != level_offsets:
        raise RuntimeError("classifier_head.pt level_offsets mismatch.")
    model.level_heads.load_state_dict(classifier_state["level_heads"])
    model.global_head.load_state_dict(classifier_state["global_head"])
    model.post_ln.load_state_dict(classifier_state["post_ln"])
    model.has_trained_local_heads = has_trained_local_heads
    model.has_trained_global_head = has_trained_global_head
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard_size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.attention_topk <= 0:
        raise ValueError("--attention_topk must be positive")
    knowledge_types = parse_knowledge_types(args.knowledge_types)
    rank, world_size, local_rank = distributed_context()
    seed_everything(args.seed + rank)
    device = torch.device("cuda", local_rank)

    checkpoint_dir = os.path.abspath(
        args.checkpoint_dir or DEFAULT_TEACHER_CHECKPOINTS[args.dataset]
    )
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    teacher_config = find_teacher_config(checkpoint_dir)
    if teacher_config.get("dataset") != args.dataset:
        raise RuntimeError(
            f"Checkpoint dataset={teacher_config.get('dataset')} but --dataset={args.dataset}."
        )
    shared_metadata: list[Any] = [None, None]
    if rank == 0:
        shared_metadata[0] = teacher_checkpoint_fingerprint(checkpoint_dir)
        shared_metadata[1] = read_head_training_status(
            checkpoint_dir,
            teacher_config,
        )
    if dist.is_initialized():
        dist.broadcast_object_list(shared_metadata, src=0)
    checkpoint_fingerprint_data = shared_metadata[0]
    head_training_status = shared_metadata[1]
    if not isinstance(checkpoint_fingerprint_data, dict):
        raise RuntimeError("Failed to obtain teacher checkpoint fingerprint.")
    if not isinstance(head_training_status, dict):
        raise RuntimeError("Failed to obtain teacher head training status.")
    teacher_config["model_path"] = checkpoint_fingerprint_data["base_model_path"]
    checkpoint_fingerprint = str(checkpoint_fingerprint_data["sha256"])
    if (
        "logits" in knowledge_types
        and not bool(head_training_status["has_trained_global_head"])
    ):
        raise RuntimeError(
            "Logit extraction requires a teacher with a trained global head."
        )

    dataset_cfg = DATASET_ZOO[args.dataset]
    data_dir = args.data_dir or dataset_cfg["path"]
    train_path = os.path.join(data_dir, dataset_cfg["train_file"])
    meta = load_preprocess_meta(train_path)
    if int(teacher_config["num_levels"]) != meta["num_levels"]:
        raise RuntimeError("teacher_config num_levels differs from train metadata.")
    expected_mapping = normalize_layer_label2id(
        {"layer_label2id": teacher_config["layer_label2id"]}
    )
    current_mapping = normalize_layer_label2id(meta["meta"])
    if expected_mapping != current_mapping:
        raise RuntimeError("Teacher label mapping differs from train metadata.")

    num_levels = meta["num_levels"]
    level_classes = meta["level_classes"]
    level_offsets = build_level_offsets(level_classes)
    total_classes = meta["total_classes"]
    max_length = int(teacher_config.get("max_length", dataset_cfg["max_length"]))
    truncate_mode = str(teacher_config.get("truncate_mode", "head"))
    feature_layer = configured_feature_layer(teacher_config)

    tokenizer_from_checkpoint = os.path.exists(
        os.path.join(checkpoint_dir, "tokenizer_config.json")
    )
    tokenizer_path = (
        checkpoint_dir if tokenizer_from_checkpoint else teacher_config["model_path"]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not tokenizer_from_checkpoint:
        add_level_special_tokens(tokenizer, LEVEL_TOKENS[:num_levels])
    level_token_ids = tokenizer.convert_tokens_to_ids(LEVEL_TOKENS[:num_levels])
    expected_level_ids = list(teacher_config.get("level_token_ids", []))
    if expected_level_ids and expected_level_ids != level_token_ids:
        raise RuntimeError(
            f"Level token ids mismatch: checkpoint={expected_level_ids}, actual={level_token_ids}"
        )

    dataset = HierarchicalDataset(
        train_path,
        tokenizer,
        num_levels,
        total_classes,
        level_offsets,
        max_length,
        truncate_mode,
        strict=True,
    )
    identities = build_sample_identities(
        train_path,
        args.dataset,
        "train",
        num_levels,
    )
    if len(identities) != len(dataset):
        raise RuntimeError(
            f"Identity count {len(identities)} differs from dataset count {len(dataset)}."
        )

    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(checkpoint_dir, "teacher_knowledge_train")
    )
    manifest_path = os.path.join(output_dir, "manifest.json")
    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    data_sha256 = sha256_file(train_path)
    num_shards = (len(dataset) + args.shard_size - 1) // args.shard_size
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "extracting",
        "dataset": args.dataset,
        "split": "train",
        "data_path": os.path.abspath(train_path),
        "data_sha256": data_sha256,
        "num_samples": len(dataset),
        "shard_size": args.shard_size,
        "num_shards": num_shards,
        "checkpoint_dir": checkpoint_dir,
        "teacher_checkpoint_fingerprint": checkpoint_fingerprint,
        "teacher_checkpoint_files": checkpoint_fingerprint_data["files"],
        "teacher_model_path": checkpoint_fingerprint_data["base_model_path"],
        "teacher_hidden_size": None,
        "num_levels": num_levels,
        "level_classes": level_classes,
        "level_offsets": level_offsets,
        "total_classes": total_classes,
        "label_space_source": meta["meta"].get("label_space_source"),
        "layer_label2id": normalize_layer_label2id(meta["meta"]),
        "level_tokens": LEVEL_TOKENS[:num_levels],
        "teacher_level_token_ids": level_token_ids,
        "tokenizer_path": tokenizer_path,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_sha256": tokenizer_hash,
        "max_length": max_length,
        "truncate_mode": truncate_mode,
        "knowledge_types": knowledge_types,
        "has_trained_local_heads": bool(
            head_training_status["has_trained_local_heads"]
        ),
        "has_trained_global_head": bool(
            head_training_status["has_trained_global_head"]
        ),
        "feature_type": "post_ln",
        "feature_layer": feature_layer,
        "attention_layer": feature_layer,
        "attention_head_reduction": "mean",
        "attention_scope": "level_token_to_query_tokens",
        "attention_topk": args.attention_topk,
        "attention_distribution": "text_renormalized_topk_plus_other",
        "seed": args.seed,
        "created_at_unix": int(time.time()),
        "world_size_at_creation": world_size,
    }

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(manifest_path):
            existing = read_json(manifest_path)
            immutable_checks = {
                "dataset": args.dataset,
                "split": "train",
                "data_sha256": data_sha256,
                "num_samples": len(dataset),
                "shard_size": args.shard_size,
                "checkpoint_dir": checkpoint_dir,
                "teacher_checkpoint_fingerprint": checkpoint_fingerprint,
                "tokenizer_sha256": tokenizer_hash,
                "max_length": max_length,
                "truncate_mode": truncate_mode,
                "attention_topk": args.attention_topk,
                "feature_layer": feature_layer,
                "attention_layer": feature_layer,
            }
            mismatches = [
                f"{key}: existing={existing.get(key)!r}, current={value!r}"
                for key, value in immutable_checks.items()
                if existing.get(key) != value
            ]
            if mismatches:
                raise RuntimeError(
                    "Existing knowledge manifest is incompatible:\n"
                    + "\n".join(mismatches)
                )
            merged_types = sorted(
                set(existing.get("knowledge_types", [])) | set(knowledge_types)
            )
            manifest = {**existing, **manifest, "knowledge_types": merged_types}
        atomic_json(manifest_path, manifest)
    barrier()

    need_attention = "attention" in knowledge_types
    model = load_teacher(
        checkpoint_dir,
        teacher_config,
        tokenizer,
        need_attention,
        device,
    )
    if bool(model.has_trained_local_heads) != bool(
        head_training_status["has_trained_local_heads"]
    ):
        raise RuntimeError("Loaded teacher local-head status changed unexpectedly.")
    if bool(model.has_trained_global_head) != bool(
        head_training_status["has_trained_global_head"]
    ):
        raise RuntimeError("Loaded teacher global-head status changed unexpectedly.")
    manifest["teacher_hidden_size"] = int(model.backbone.config.hidden_size)
    attention_capture = (
        LayerAttentionCapture(
            model.backbone,
            layer_idx=feature_layer,
        )
        if need_attention
        else None
    )
    if attention_capture is not None:
        manifest["attention_layer_resolved"] = (
            attention_capture.resolved_layer_idx
        )
        manifest["num_decoder_layers"] = attention_capture.num_decoder_layers
    attention_validation: dict[str, float] | None = None
    force_full_attention_fallback = False
    if attention_capture is not None:
        validation_sample = dataset[0]
        validation_input_ids = validation_sample["input_ids"].unsqueeze(0).to(device)
        validation_attention_mask = (
            validation_sample["attention_mask"].unsqueeze(0).to(device)
        )
        local_validation_error: str | None = None
        try:
            attention_validation = validate_attention_replay(
                model.backbone,
                attention_capture,
                validation_input_ids,
                validation_attention_mask,
                num_levels,
            )
        except RuntimeError as exc:
            local_validation_error = str(exc)
        failure_flag = torch.tensor(
            [1 if local_validation_error is not None else 0],
            dtype=torch.int32,
            device=device,
        )
        if dist.is_initialized():
            dist.all_reduce(failure_flag, op=dist.ReduceOp.MAX)
        any_rank_failed = bool(failure_flag.item())
        if any_rank_failed:
            if not args.allow_full_attention_fallback:
                detail = (
                    local_validation_error
                    or "attention replay validation failed on another rank"
                )
                raise RuntimeError(
                    "At least one rank failed attention replay validation: "
                    f"{detail}"
                )
            force_full_attention_fallback = True
            attention_capture.close()
            attention_capture = None
            attention_validation = None
            if rank == 0:
                print(
                    "At least one rank failed attention replay validation; all "
                    "ranks will use the explicitly enabled full-attention fallback."
                )
        elif rank == 0:
            assert attention_validation is not None
            print(
                "Attention replay validation passed on every rank: "
                f"seq_len={int(attention_validation['validation_seq_len'])}, "
                f"max_abs_error={attention_validation['max_abs_error']:.8f}, "
                f"mean_abs_error={attention_validation['mean_abs_error']:.8f}"
            )

    progress = tqdm(
        range(rank, num_shards, world_size),
        disable=rank != 0,
        desc="Teacher shards",
    )
    for shard_id in progress:
        start_idx = shard_id * args.shard_size
        end_idx = min(len(dataset), start_idx + args.shard_size)
        expected_shard_uids = uid_hex_to_tensor(
            [
                identity["sample_uid"]
                for identity in identities[start_idx:end_idx]
            ]
        )
        requested_complete = True
        for module_name in ["sample_index", *knowledge_types]:
            tensor_path, sidecar_path = shard_paths(
                output_dir,
                module_name,
                shard_id,
            )
            valid = validate_existing_shard(
                tensor_path,
                sidecar_path,
                start_idx,
                end_idx,
                module_name,
                manifest,
                expected_shard_uids,
                checkpoint_fingerprint,
            )
            if args.overwrite_shards or not valid:
                requested_complete = False
                break
        if requested_complete:
            continue

        subset = Subset(dataset, range(start_idx, end_idx))
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=trim_right_padding_collator,
        )
        accum: dict[str, list[torch.Tensor]] = {
            "dataset_idx": [],
            "source_line_no": [],
            "sample_uid": [],
            "query_hash": [],
            "teacher_global_logits": [],
            "teacher_local_logits": [],
            "teacher_level_features": [],
            "topk_positions": [],
            "topk_token_ids": [],
            "topk_valid_mask": [],
            "teacher_topk_attention_mean": [],
            "teacher_other_attention_mean": [],
            "teacher_text_attention_mass": [],
            "topk_char_offsets": [],
        }
        for batch in loader:
            batch_indices = batch["dataset_idx"].tolist()
            batch_identities = [identities[index] for index in batch_indices]
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            if attention_capture is not None:
                attention_capture.clear()

            with torch.inference_mode():
                outputs, post_ln_features = extract_model_outputs(
                    model,
                    input_ids,
                    attention_mask,
                )
                if "logits" in knowledge_types:
                    accum["teacher_global_logits"].append(
                        outputs["global_logits"].to(torch.float16).cpu()
                    )
                    if model.has_trained_local_heads:
                        local_logits = torch.cat(
                            outputs["logits_list"],
                            dim=-1,
                        )
                        accum["teacher_local_logits"].append(
                            local_logits.to(torch.float16).cpu()
                        )
                if "features" in knowledge_types:
                    accum["teacher_level_features"].append(
                        post_ln_features.to(torch.float16).cpu()
                    )
                if need_attention:
                    if force_full_attention_fallback:
                        fallback = model.backbone(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_attentions=True,
                            return_dict=True,
                        )
                        if not fallback.attentions:
                            raise RuntimeError(
                                "Full attention fallback returned no attentions."
                            )
                        last_attention = fallback.attentions[
                            int(manifest["attention_layer_resolved"])
                        ]
                    else:
                        assert attention_capture is not None
                        last_attention = attention_capture.replay()
                    queries = [dataset.data[index]["query"] for index in batch_indices]
                    offsets = encode_offsets(
                        tokenizer,
                        queries,
                        input_ids,
                        attention_mask,
                        num_levels,
                        truncate_mode,
                    )
                    attention_payload = topk_attention_knowledge(
                        last_attention,
                        input_ids,
                        attention_mask,
                        level_token_ids,
                        args.attention_topk,
                        offsets,
                    )
                    for key, value in attention_payload.items():
                        accum[key].append(value.cpu())

            accum["dataset_idx"].append(
                torch.tensor(batch_indices, dtype=torch.int64)
            )
            accum["source_line_no"].append(
                torch.tensor(
                    [item["source_line_no"] for item in batch_identities],
                    dtype=torch.int64,
                )
            )
            accum["sample_uid"].append(
                uid_hex_to_tensor([item["sample_uid"] for item in batch_identities])
            )
            accum["query_hash"].append(
                uid_hex_to_tensor([item["query_hash"] for item in batch_identities])
            )

        combined = {
            key: torch.cat(values, dim=0)
            for key, values in accum.items()
            if values
        }
        expected_indices = torch.arange(start_idx, end_idx, dtype=torch.int64)
        if not torch.equal(combined["dataset_idx"], expected_indices):
            raise RuntimeError(
                f"Shard {shard_id} dataset indices are not contiguous and ordered."
            )
        common = {
            "dataset_idx": combined["dataset_idx"],
            "sample_uid": combined["sample_uid"],
        }
        sample_payload = {
            "source_line_no": combined["source_line_no"],
            "query_hash": combined["query_hash"],
        }
        save_module_shard(
            output_dir,
            "sample_index",
            shard_id,
            start_idx,
            end_idx,
            common,
            sample_payload,
            checkpoint_fingerprint,
        )
        if "logits" in knowledge_types:
            logits_payload = {
                "teacher_global_logits": combined["teacher_global_logits"],
            }
            if model.has_trained_local_heads:
                logits_payload["teacher_local_logits"] = combined[
                    "teacher_local_logits"
                ]
            save_module_shard(
                output_dir,
                "logits",
                shard_id,
                start_idx,
                end_idx,
                common,
                logits_payload,
                checkpoint_fingerprint,
            )
        if "features" in knowledge_types:
            save_module_shard(
                output_dir,
                "features",
                shard_id,
                start_idx,
                end_idx,
                common,
                {
                    "teacher_level_features": combined[
                        "teacher_level_features"
                    ],
                },
                checkpoint_fingerprint,
            )
        if "attention" in knowledge_types:
            save_module_shard(
                output_dir,
                "attention",
                shard_id,
                start_idx,
                end_idx,
                common,
                {
                    key: combined[key]
                    for key in (
                        "topk_positions",
                        "topk_token_ids",
                        "topk_valid_mask",
                        "teacher_topk_attention_mean",
                        "teacher_other_attention_mean",
                        "teacher_text_attention_mass",
                        "topk_char_offsets",
                    )
                },
                checkpoint_fingerprint,
            )

    if attention_capture is not None:
        attention_capture.close()
    barrier()

    if rank == 0:
        final_manifest = read_json(manifest_path)
        final_manifest["teacher_hidden_size"] = int(
            model.backbone.config.hidden_size
        )
        final_manifest["has_trained_local_heads"] = bool(
            model.has_trained_local_heads
        )
        final_manifest["has_trained_global_head"] = bool(
            model.has_trained_global_head
        )
        if need_attention:
            final_manifest["attention_layer"] = feature_layer
            final_manifest["attention_layer_resolved"] = (
                manifest["attention_layer_resolved"]
            )
            final_manifest["num_decoder_layers"] = manifest[
                "num_decoder_layers"
            ]
        completed_modules = ["sample_index", *final_manifest["knowledge_types"]]
        missing: list[str] = []
        for module_name in completed_modules:
            for shard_id in range(num_shards):
                start_idx = shard_id * args.shard_size
                end_idx = min(len(dataset), start_idx + args.shard_size)
                tensor_path, sidecar_path = shard_paths(
                    output_dir,
                    module_name,
                    shard_id,
                )
                expected_shard_uids = uid_hex_to_tensor(
                    [
                        identity["sample_uid"]
                        for identity in identities[start_idx:end_idx]
                    ]
                )
                if not validate_existing_shard(
                    tensor_path,
                    sidecar_path,
                    start_idx,
                    end_idx,
                    module_name,
                    final_manifest,
                    expected_shard_uids,
                    checkpoint_fingerprint,
                ):
                    missing.append(f"{module_name}/shard-{shard_id:05d}")
        if missing:
            final_manifest["status"] = "incomplete"
            final_manifest["invalid_or_missing_shards"] = missing
            atomic_json(manifest_path, final_manifest)
            raise RuntimeError(
                f"Extraction incomplete; {len(missing)} invalid/missing shards. "
                f"See {manifest_path}"
            )
        final_manifest["status"] = "complete"
        if attention_validation is not None:
            final_manifest["attention_replay_validation"] = attention_validation
        if need_attention:
            final_manifest["attention_extraction_mode"] = (
                "full_forward_fallback"
                if force_full_attention_fallback
                else "validated_selected_layer_replay"
            )
        final_manifest["completed_at_unix"] = int(time.time())
        final_manifest.pop("invalid_or_missing_shards", None)
        atomic_json(manifest_path, final_manifest)
        print(f"Teacher extraction complete: {output_dir}")

    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
