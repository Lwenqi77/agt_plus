"""
Cached level-token Top-k KD, copied from the user's train_distill_mi_ma.py.
Only the attention reader/target construction, CLI and provenance are changed.
DistillationTrainer (including losses), model, optimizer and original defaults
are unchanged. See CACHED_TOPK_DISTILLATION.md for commands and validation.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from convert_full_attention_to_level_text import (
    MATCH_KEYS as CACHED_MATCH_KEYS,
    FORMAT as CACHED_FORMAT,
    read_light_manifest,
    read_light_shard,
)

from extract_teacher import (
    SCHEMA_VERSION,
    build_sample_identities,
    find_decoder_layers,
    is_attention_implementation_compat_error,
    read_json,
    sha256_file,
    tensor_uid_hex,
    tokenizer_fingerprint,
    uid_hex_to_tensor,
    validate_existing_shard,
)
from train_baseline import (
    DATASET_ZOO,
    LEVEL_TOKENS,
    MODEL_ZOO,
    HierarchicalDataset,
    HierarchicalTeacher,
    TeeLogger,
    add_level_special_tokens,
    assert_preprocess_meta_compatible,
    build_level_offsets,
    load_preprocess_meta,
    masked_multilabel_f1,
    normalize_layer_label2id,
    seed_everything,
    write_json,
)


def parse_cached_topk(value):
    if value == "all":
        return value
    try:
        k = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("attention_topk must be a positive integer or all") from exc
    if k < 1:
        raise argparse.ArgumentTypeError("attention_topk must be positive")
    return k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a hierarchical student from offline teacher knowledge."
    )
    parser.add_argument("--dataset", default="wos", choices=sorted(DATASET_ZOO))
    parser.add_argument("--knowledge_dir", required=True)
    parser.add_argument("--attention_cache_dir", help="Compact FP32 level-to-text bank; required when attention KD is enabled")
    parser.add_argument("--attention_topk", type=parse_cached_topk, default=32,
                        help="Positive k or all; teacher selections made from compact FP32 rows")
    parser.add_argument("--attention_cache_samples", type=int, default=128,
                        help="Maximum cached sparse targets per loader process; 0 disables this LRU cache")
    parser.add_argument("--model_key", default="qwen0.6b", choices=sorted(MODEL_ZOO))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_name", default=None)

    parser.add_argument("--tuning_mode", default="lora", choices=["lora", "full"])
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--global_supervised_bce_weight", type=float, default=1.0)
    parser.add_argument("--local_supervised_bce_weight", type=float, default=0.0)
    parser.add_argument("--global_logit_kd_weight", type=float, default=1.0)
    parser.add_argument("--local_logit_kd_weight", type=float, default=0.0)
    parser.add_argument("--feature_kd_weight", type=float, default=0.0)
    parser.add_argument("--feature_cosine_weight", type=float, default=1.0)
    parser.add_argument("--attention_kd_weight", type=float, default=0.0)
    parser.add_argument(
        "--attention_mass_kd_weight",
        type=float,
        default=0.0,
        help="MSE weight for total level-token attention mass assigned to text.",
    )
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--student_feature_layer",
        type=int,
        default=-1,
        help="Student Transformer block used for classification/features; -1 is last.",
    )
    parser.add_argument(
        "--student_attention_layer",
        type=int,
        default=None,
        help="Student attention block; defaults to --student_feature_layer.",
    )

    parser.add_argument("--train_batch", type=int, default=8)
    parser.add_argument("--eval_batch", type=int, default=8)
    parser.add_argument("--grad_acc", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument(
        "--early_stopping_target",
        choices=[
            "global_macro",
            "local_macro",
            "global_supervised_bce",
            "local_supervised_bce",
            "none",
        ],
        default="global_macro",
        help="Metric used for best-checkpoint selection and early stopping.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Checkpoint directory to resume, or 'latest' under --output_dir.",
    )
    parser.add_argument("--fixed_threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_swanlab", action="store_true")
    return parser.parse_args()


def validate_loss_args(args: argparse.Namespace) -> None:
    names = [
        "global_supervised_bce_weight",
        "local_supervised_bce_weight",
        "global_logit_kd_weight",
        "local_logit_kd_weight",
        "feature_kd_weight",
        "feature_cosine_weight",
        "attention_kd_weight",
        "attention_mass_kd_weight",
    ]
    for name in names:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} cannot be negative")
    objective_weights = [
        args.global_supervised_bce_weight,
        args.local_supervised_bce_weight,
        args.global_logit_kd_weight,
        args.local_logit_kd_weight,
        args.feature_kd_weight,
        args.attention_kd_weight,
        args.attention_mass_kd_weight,
    ]
    if not any(weight > 0 for weight in objective_weights):
        raise ValueError("At least one training objective weight must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    attention_enabled = (
        args.attention_kd_weight > 0
        or args.attention_mass_kd_weight > 0
    )
    if attention_enabled and args.gradient_checkpointing:
        raise ValueError(
            "Attention KD cannot be combined with --gradient_checkpointing: "
            "selected-layer attention is captured from the original forward graph."
        )

    global_head_trained = (
        args.global_supervised_bce_weight > 0
        or args.global_logit_kd_weight > 0
    )
    local_heads_trained = (
        args.local_supervised_bce_weight > 0
        or args.local_logit_kd_weight > 0
    )
    if (
        args.early_stopping_target.startswith("global_")
        and not global_head_trained
    ):
        raise ValueError(
            "--early_stopping_target selects the global head, but no global-head "
            "training objective is enabled. Choose local_macro, "
            "local_supervised_bce, or none."
        )
    if (
        args.early_stopping_target.startswith("local_")
        and not local_heads_trained
    ):
        raise ValueError(
            "--early_stopping_target selects local heads, but no local-head "
            "training objective is enabled. Choose global_macro, "
            "global_supervised_bce, or none."
        )


def load_trusted_torch_file(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_sharded_backbone_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    prefer_safe: bool,
) -> None:
    """Strictly load a sharded full-finetuning checkpoint without private HF APIs."""
    safe_index_path = os.path.join(
        checkpoint_dir,
        "model.safetensors.index.json",
    )
    bin_index_path = os.path.join(
        checkpoint_dir,
        "pytorch_model.bin.index.json",
    )
    if prefer_safe and os.path.exists(safe_index_path):
        index_path = safe_index_path
        load_shard = lambda path: load_file(path, device="cpu")
    elif os.path.exists(bin_index_path):
        index_path = bin_index_path
        load_shard = load_trusted_torch_file
    elif os.path.exists(safe_index_path):
        index_path = safe_index_path
        load_shard = lambda path: load_file(path, device="cpu")
    else:
        raise FileNotFoundError(
            "Full-finetuning checkpoint lacks a recognized sharded index: "
            f"{checkpoint_dir}"
        )

    index = read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"Invalid checkpoint weight_map: {index_path}")

    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(weight_map.keys())
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Sharded backbone checkpoint does not strictly match the model. "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )

    for shard_name in sorted(set(weight_map.values())):
        shard_path = os.path.join(checkpoint_dir, shard_name)
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(
                f"Checkpoint shard listed by the index is missing: {shard_path}"
            )
        shard_state = load_shard(shard_path)
        shard_unexpected = set(shard_state.keys()) - checkpoint_keys
        if shard_unexpected:
            raise RuntimeError(
                f"Checkpoint shard {shard_path} has unindexed keys: "
                f"{sorted(shard_unexpected)[:20]}"
            )
        model.load_state_dict(shard_state, strict=False)
        del shard_state


def load_knowledge_module(
    knowledge_dir: str,
    module_name: str,
    manifest: dict[str, Any],
    expected_uids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {}
    for shard_id in range(int(manifest["num_shards"])):
        path = os.path.join(
            knowledge_dir,
            module_name,
            f"shard-{shard_id:05d}.safetensors",
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing knowledge shard: {path}")
        sidecar_path = os.path.join(
            knowledge_dir,
            module_name,
            f"shard-{shard_id:05d}.json",
        )
        start_idx = shard_id * int(manifest["shard_size"])
        end_idx = min(
            int(manifest["num_samples"]),
            start_idx + int(manifest["shard_size"]),
        )
        if not validate_existing_shard(
            path,
            sidecar_path,
            start_idx,
            end_idx,
            module_name,
            manifest,
            expected_uids[start_idx:end_idx],
            str(manifest["teacher_checkpoint_fingerprint"]),
        ):
            raise RuntimeError(
                f"Knowledge shard failed structural/checksum validation: {path}"
            )
        shard = load_file(path, device="cpu")
        for key, value in shard.items():
            chunks.setdefault(key, []).append(value)
    combined = {
        key: torch.cat(values, dim=0)
        for key, values in chunks.items()
    }
    expected = torch.arange(int(manifest["num_samples"]), dtype=torch.int64)
    if not torch.equal(combined["dataset_idx"].to(torch.int64), expected):
        raise RuntimeError(
            f"{module_name} shards do not contain ordered, complete dataset_idx values."
        )
    return combined


class OfflineKnowledge:
    def __init__(
        self,
        knowledge_dir: str,
        manifest: dict[str, Any],
        required_modules: set[str],
        expected_uids: list[str],
    ):
        self.manifest = manifest
        expected_uid_tensor = uid_hex_to_tensor(expected_uids)
        sample_index = load_knowledge_module(
            knowledge_dir,
            "sample_index",
            manifest,
            expected_uid_tensor,
        )
        actual_uids = tensor_uid_hex(sample_index["sample_uid"])
        if actual_uids != expected_uids:
            mismatch = next(
                (
                    idx
                    for idx, (actual, expected) in enumerate(
                        zip(actual_uids, expected_uids)
                    )
                    if actual != expected
                ),
                None,
            )
            if mismatch is None and len(actual_uids) != len(expected_uids):
                mismatch = min(len(actual_uids), len(expected_uids))
            raise RuntimeError(
                "Teacher knowledge does not match the current train JSONL. "
                f"First mismatched dataset_idx={mismatch}."
            )
        self.sample_uid = sample_index["sample_uid"]
        self.modules: dict[str, dict[str, torch.Tensor]] = {}
        available = set(manifest.get("knowledge_types", []))
        missing = required_modules - available
        if missing:
            raise RuntimeError(
                f"Knowledge manifest lacks required modules: {sorted(missing)}"
            )
        for module_name in sorted(required_modules):
            values = load_knowledge_module(
                knowledge_dir,
                module_name,
                manifest,
                expected_uid_tensor,
            )
            if not torch.equal(values["sample_uid"], self.sample_uid):
                raise RuntimeError(
                    f"{module_name} sample_uid order differs from sample_index."
                )
            self.modules[module_name] = values

    def item(self, idx: int) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        if "logits" in self.modules:
            result["teacher_global_logits"] = self.modules["logits"][
                "teacher_global_logits"
            ][idx]
            if "teacher_local_logits" in self.modules["logits"]:
                result["teacher_local_logits"] = self.modules["logits"][
                    "teacher_local_logits"
                ][idx]
        if "features" in self.modules:
            result["teacher_level_features"] = self.modules["features"][
                "teacher_level_features"
            ][idx]
        if "attention" in self.modules:
            for key in (
                "topk_positions",
                "topk_token_ids",
                "topk_valid_mask",
                "teacher_topk_attention_mean",
                "teacher_other_attention_mean",
                "teacher_text_attention_mass",
            ):
                result[key] = self.modules["attention"][key][idx]
        return result


class DistillTrainDataset(Dataset):
    def __init__(
        self,
        base_dataset: HierarchicalDataset,
        knowledge: OfflineKnowledge,
    ):
        if len(base_dataset) != int(knowledge.manifest["num_samples"]):
            raise RuntimeError("Training dataset and knowledge sample counts differ.")
        self.base_dataset = base_dataset
        self.knowledge = knowledge

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            **self.base_dataset[idx],
            **self.knowledge.item(idx),
        }


def stack_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
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


def level_text_topk(raw_attention, input_ids, level_ids, topk):
    """Reproduce legacy per-level FP32 normalize/topk -> FP16 target storage.

    Input is post-softmax raw head mean, not a pre-normalized Top-k subset.
    Selection runs on CPU. Ties/reductions can differ from old GPU extraction;
    this does not claim bitwise equality to a separately extracted GPU bank.
    """
    levels = len(level_ids)
    length = int(input_ids.numel()) - levels
    if (isinstance(topk, bool) or not isinstance(topk, int) or topk < 1
            or raw_attention.dtype != torch.float32 or raw_attention.shape != (levels, length)
            or length < 1 or input_ids.ndim != 1):
        raise ValueError("Invalid FP32 compact attention shape or top-k")
    if input_ids[-levels:].tolist() != list(level_ids) or any(bool((input_ids[:length] == i).any()) for i in level_ids):
        raise ValueError("Compact attention/token layout mismatch")
    if not bool(torch.isfinite(raw_attention).all()) or bool((raw_attention < 0).any()):
        raise ValueError("Nonfinite or negative compact attention")
    positions = torch.full((levels, topk), -1, dtype=torch.int32)
    token_ids = torch.full_like(positions, -1)
    valid = torch.zeros(levels, topk, dtype=torch.bool)
    weights = torch.zeros(levels, topk, dtype=torch.float32)
    other, masses = torch.ones(levels, dtype=torch.float32), torch.zeros(levels, dtype=torch.float32)
    for level in range(levels):
        scores = raw_attention[level].clamp_min(0)
        mass = scores.sum()
        if not torch.isfinite(mass) or mass.item() <= 0:
            raise ValueError("Nonpositive compact text attention mass")
        normalized = scores / mass
        selected, selected_positions = torch.topk(normalized, min(topk, length), largest=True, sorted=True)
        k = selected.numel()
        positions[level, :k] = selected_positions.to(torch.int32)
        token_ids[level, :k] = input_ids[selected_positions].to(torch.int32)
        valid[level, :k] = True
        weights[level, :k] = selected
        other[level] = (1.0 - selected.sum()).clamp_min(0)
        masses[level] = mass
    return {"topk_positions": positions, "topk_token_ids": token_ids, "topk_valid_mask": valid,
            "teacher_topk_attention_mean": weights.to(torch.float16),
            "teacher_other_attention_mean": other.to(torch.float16),
            "teacher_text_attention_mass": masses.to(torch.float16)}


class CachedLevelTextKnowledge:
    def __init__(self, root, reference, identities, topk=32, cache_samples=128):
        self.root = Path(root).resolve()
        self.manifest = read_light_manifest(self.root)
        m = self.manifest
        for key in CACHED_MATCH_KEYS:
            if key not in reference or reference[key] != m[key]:
                raise ValueError(f"Compact attention and original knowledge mismatch: {key}")
        self.layer = int(m["teacher_layer"])
        feature_layer = int(reference["feature_layer"])
        if feature_layer < 0:
            feature_layer += int(m["num_decoder_layers"])
        if self.layer != feature_layer:
            raise ValueError("Teacher attention layer differs from its classification feature layer")
        self.requested_topk = topk
        self.topk = int(m["max_length"]) - int(m["num_levels"]) if topk == "all" else topk
        if (isinstance(self.topk, bool) or not isinstance(self.topk, int) or self.topk < 1
                or cache_samples < 0 or len(identities) != m["num_samples"]):
            raise ValueError("Invalid top-k, sample count or sparse cache size")
        self.cache_limit, self.cache = cache_samples, OrderedDict()
        self.headers, self.file_stats = [], []
        for shard in range(int(m["num_shards"])):
            values = read_light_shard(self.root, m, shard)
            start = shard * m["shard_size"]
            rows = identities[start:start + len(values["dataset_idx"])]
            expected_uids = np.array([list(bytes.fromhex(r["sample_uid"])) for r in rows], dtype=np.uint8)
            expected_hashes = np.array([list(bytes.fromhex(r["query_hash"])) for r in rows], dtype=np.uint8)
            if (not np.array_equal(values["sample_uid"], expected_uids)
                    or not np.array_equal(values["query_hash"], expected_hashes)
                    or not np.array_equal(values["source_line_no"], [r["source_line_no"] for r in rows])):
                raise ValueError(f"Compact sample identities mismatch in shard {shard}")
            # Retain only tiny offsets, not entire FP32 shards, for shuffled I/O.
            self.headers.append({k: values[k].copy() for k in ("text_lengths", "token_offsets", "row_offsets")})
            stat = self._path(shard).stat()
            self.file_stats.append((stat.st_size, stat.st_mtime_ns))
        del values

    def _path(self, shard):
        return self.root / "attention" / f"shard-{shard:05d}.safetensors"

    def compatibility(self):
        return {"bank_fingerprint": self.manifest["bank_fingerprint"],
                "source_bank_fingerprint": self.manifest["source_bank_fingerprint"],
                "teacher_layer": self.layer, "storage_dtype": "float32",
                "requested_topk": self.requested_topk, "effective_topk": self.topk,
                "target_construction": "cpu_fp32_text_normalization_torch_topk_then_fp16_topk_other_mass_v1",
                "student_and_loss": "unchanged_original_level_token_bins_KL_and_masked_mean"}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["cache"] = OrderedDict()
        return state

    def item(self, index):
        if not 0 <= index < self.manifest["num_samples"]:
            raise IndexError(index)
        shard, row = divmod(index, self.manifest["shard_size"])
        path = self._path(shard)
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.file_stats[shard]:
            raise ValueError("Compact shard changed after validation")
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        offsets = self.headers[shard]
        with safe_open(str(path), framework="np", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("format") != CACHED_FORMAT or metadata.get("bank_fingerprint") != self.manifest["bank_fingerprint"]:
                raise ValueError("Compact tensor provenance changed")
            begin, end = map(int, offsets["row_offsets"][row:row+2])
            raw = handle.get_slice("teacher_text_attention_raw_mean")[begin:end]
            raw = torch.from_numpy(raw.copy()).reshape(self.manifest["num_levels"], int(offsets["text_lengths"][row]))
            begin, end = map(int, offsets["token_offsets"][row:row+2])
            ids = torch.from_numpy(handle.get_slice("input_ids")[begin:end].copy())
        result = level_text_topk(raw, ids, self.manifest["teacher_level_token_ids"], self.topk)
        result["teacher_cached_input_ids"] = ids
        if self.cache_limit:
            while len(self.cache) >= self.cache_limit:
                self.cache.popitem(last=False)
            self.cache[index] = result
        return result


class CombinedCachedKnowledge:
    def __init__(self, original, attention):
        self.original, self.attention, self.manifest = original, attention, original.manifest

    def item(self, index):
        return {**self.original.item(index), **self.attention.item(index)}


def cached_topk_collator(features):
    cleaned = []
    for source in features:
        row = dict(source)
        cached = row.pop("teacher_cached_input_ids", None)
        if cached is not None:
            mask = row["attention_mask"]
            length = int(mask.sum())
            if not bool((mask[:length] == 1).all()) or bool(mask[length:].any()):
                raise ValueError("Cached attention requires contiguous right padding")
            if not torch.equal(cached.long(), row["input_ids"][:length].long()):
                raise ValueError(f"Compact teacher/student input IDs differ at sample {row.get('dataset_idx')}")
        cleaned.append(row)
    return stack_collator(cleaned)


class SameForwardLayerAttentionCapture:
    """Capture selected-layer attention from the student's original forward."""

    def __init__(self, backbone: nn.Module, layer_idx: int = -1):
        layers = find_decoder_layers(backbone)
        resolved = layer_idx if layer_idx >= 0 else len(layers) + layer_idx
        if not 0 <= resolved < len(layers):
            raise ValueError(
                f"Student attention layer {layer_idx} resolves to {resolved}, "
                f"outside [0, {len(layers)})."
            )
        self.layer_idx = int(layer_idx)
        self.resolved_layer_idx = int(resolved)
        self.num_decoder_layers = len(layers)
        self.self_attn = layers[resolved].self_attn
        self.enabled = False
        self.attention: torch.Tensor | None = None
        self._pre_hook_with_kwargs = False
        try:
            self.pre_hook = self.self_attn.register_forward_pre_hook(
                self._enable_attention_output,
                with_kwargs=True,
            )
            self._pre_hook_with_kwargs = True
        except TypeError:
            self.pre_hook = None
        try:
            self.output_hook = self.self_attn.register_forward_hook(
                self._capture_output_with_kwargs,
                with_kwargs=True,
            )
            self._output_hook_with_kwargs = True
        except TypeError:
            self.output_hook = self.self_attn.register_forward_hook(
                self._capture_output_args_only
            )
            self._output_hook_with_kwargs = False

    def clear(self) -> None:
        self.attention = None

    def close(self) -> None:
        if self.pre_hook is not None:
            self.pre_hook.remove()
        self.output_hook.remove()

    def _enable_attention_output(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        if not self.enabled:
            return None
        signature = inspect.signature(module.forward)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_var_kwargs or "output_attentions" in signature.parameters:
            updated = dict(kwargs)
            updated["output_attentions"] = True
            return args, updated
        return None

    @staticmethod
    def _find_attention(
        output: Any,
        seq_len: int,
    ) -> torch.Tensor:
        candidates: list[torch.Tensor] = []
        if torch.is_tensor(output):
            candidates.append(output)
        elif isinstance(output, (tuple, list)):
            candidates.extend(
                value for value in output if torch.is_tensor(value)
            )
        else:
            for name in ("attn_weights", "attention_weights", "attentions"):
                value = getattr(output, name, None)
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
            "The selected student self-attention layer did not expose a "
            "[batch,heads,seq,seq] tensor during its original forward. "
            f"Tensor outputs were: {shapes}"
        )

    def _capture(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        if not self.enabled:
            return
        hidden_states = kwargs.get("hidden_states")
        if not torch.is_tensor(hidden_states) and args:
            hidden_states = args[0]
        if not torch.is_tensor(hidden_states):
            raise RuntimeError(
                "Could not identify hidden_states for selected student attention."
            )
        self.attention = self._find_attention(
            output,
            int(hidden_states.shape[1]),
        )

    def _capture_output_with_kwargs(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        self._capture(args, kwargs, output)

    def _capture_output_args_only(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
        output: Any,
    ) -> None:
        self._capture(args, {}, output)

    def get(self) -> torch.Tensor:
        if self.attention is None:
            raise RuntimeError(
                "Selected student attention was not captured from the original "
                "forward. Ensure eager attention is active."
            )
        return self.attention


def validate_same_forward_attention_capture(
    model: nn.Module,
    capture: SameForwardLayerAttentionCapture,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_levels: int,
    max_query_tokens: int = 64,
) -> dict[str, float]:
    valid_len = int(attention_mask[0].sum().item())
    query_len = valid_len - num_levels
    if query_len <= 0:
        raise RuntimeError("Attention validation sample has no query tokens.")
    kept_query_len = min(query_len, max_query_tokens)
    short_ids = torch.cat(
        [
            input_ids[0, :kept_query_len],
            input_ids[0, query_len:valid_len],
        ],
        dim=0,
    ).unsqueeze(0)
    short_mask = torch.ones_like(short_ids)
    was_training = model.training
    model.eval()
    capture.clear()
    capture.enabled = True
    try:
        with torch.inference_mode():
            model(input_ids=short_ids, attention_mask=short_mask)
            attention = capture.get().float()
    finally:
        capture.enabled = False
        model.train(was_training)
    if not bool(torch.isfinite(attention).all().item()):
        raise RuntimeError("Captured student attention contains NaN/Inf.")
    row_sum_error = (attention.sum(dim=-1) - 1.0).abs()
    max_row_sum_error = float(row_sum_error.max().item())
    if max_row_sum_error > 5e-3:
        raise RuntimeError(
            "Captured student attention rows are not normalized: "
            f"max_row_sum_error={max_row_sum_error:.8f}."
        )
    return {
        "validation_seq_len": float(short_ids.shape[1]),
        "max_row_sum_error": max_row_sum_error,
        "mean_row_sum_error": float(row_sum_error.mean().item()),
    }


class DistillStudent(nn.Module):
    def __init__(
        self,
        core: HierarchicalTeacher,
        teacher_hidden_size: int,
        tuning_mode: str,
        enable_feature_projection: bool,
    ):
        super().__init__()
        self.core = core
        self.tuning_mode = tuning_mode
        self.enable_feature_projection = enable_feature_projection
        student_hidden_size = int(core.backbone.config.hidden_size)
        self.feature_projection = nn.Linear(
            student_hidden_size,
            teacher_hidden_size,
            bias=False,
        )
        target_dtype = getattr(core.backbone, "dtype", torch.float32)
        self.feature_projection.to(target_dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        outputs = self.core(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        outputs["post_ln_level_features"] = torch.stack(
            [
                self.core.post_ln(feature)
                for feature in outputs["level_features"]
            ],
            dim=1,
        )
        if self.enable_feature_projection:
            outputs["projected_level_features"] = self.feature_projection(
                outputs["post_ln_level_features"]
            )
        return outputs

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.core.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self) -> None:
        self.core.gradient_checkpointing_disable()


def build_flat_targets(
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    level_classes: list[int],
    level_offsets: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = []
    masks = []
    for level_idx, width in enumerate(level_classes):
        start = level_offsets[level_idx]
        end = start + width
        targets.append(labels[:, level_idx, start:end].float())
        masks.append(
            label_mask[:, level_idx]
            .float()
            .unsqueeze(1)
            .expand(-1, width)
        )
    return torch.cat(targets, dim=-1), torch.cat(masks, dim=-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class DistillationTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        tokenizer_for_save: Any,
        distill_config: dict[str, Any],
        level_token_ids: list[int],
        level_classes: list[int],
        level_offsets: list[int],
        global_supervised_bce_weight: float,
        local_supervised_bce_weight: float,
        global_logit_kd_weight: float,
        local_logit_kd_weight: float,
        feature_kd_weight: float,
        feature_cosine_weight: float,
        attention_kd_weight: float,
        attention_mass_kd_weight: float,
        student_attention_layer: int,
        temperature: float,
        fixed_threshold: float,
        early_stopping_patience: int,
        early_stopping_target: str,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.tokenizer_for_save = tokenizer_for_save
        self.distill_config = distill_config
        self.level_token_ids = [int(value) for value in level_token_ids]
        self.level_classes = [int(value) for value in level_classes]
        self.level_offsets = [int(value) for value in level_offsets]
        self.global_supervised_bce_weight = global_supervised_bce_weight
        self.local_supervised_bce_weight = local_supervised_bce_weight
        self.global_logit_kd_weight = global_logit_kd_weight
        self.local_logit_kd_weight = local_logit_kd_weight
        self.feature_kd_weight = feature_kd_weight
        self.feature_cosine_weight = feature_cosine_weight
        self.attention_kd_weight = attention_kd_weight
        self.attention_mass_kd_weight = attention_mass_kd_weight
        self.temperature = temperature
        self.fixed_threshold = fixed_threshold
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_target = early_stopping_target
        self.best_selection_score: float | None = None
        # 仅用于额外保存 best global micro，不参与 early stopping
        self.best_global_micro_score: float | None = None
        self.early_stop_bad_count = 0
        attention_enabled = (
            attention_kd_weight > 0 or attention_mass_kd_weight > 0
        )
        self.attention_capture = (
            SameForwardLayerAttentionCapture(
                self.model.core.backbone,
                layer_idx=student_attention_layer,
            )
            if attention_enabled
            else None
        )

    def _attention_bins(
        self,
        attention: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        topk_positions: torch.Tensor,
        topk_token_ids: torch.Tensor,
        topk_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean_attention = attention.float().mean(dim=1)
        batch_size, num_levels, topk = topk_positions.shape
        result = torch.zeros(
            batch_size,
            num_levels,
            topk + 1,
            dtype=torch.float32,
            device=input_ids.device,
        )
        text_mass = torch.zeros(
            batch_size,
            num_levels,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for batch_idx in range(batch_size):
            valid_len = int(attention_mask[batch_idx].sum().item())
            query_len = valid_len - num_levels
            if query_len <= 0:
                raise RuntimeError("Student sample has no query tokens.")
            for level_idx, level_token_id in enumerate(self.level_token_ids):
                matches = (
                    input_ids[batch_idx, :valid_len] == level_token_id
                ).nonzero(as_tuple=True)[0]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one student level token {level_token_id}, got {len(matches)}."
                    )
                query_position = int(matches[0].item())
                text_scores = mean_attention[
                    batch_idx,
                    query_position,
                    :query_len,
                ].clamp_min(0)
                mass = text_scores.sum()
                text_mass[batch_idx, level_idx] = mass
                text_scores = text_scores / mass.clamp_min(1e-12)
                valid = topk_valid_mask[batch_idx, level_idx].bool()
                selected_positions = topk_positions[
                    batch_idx,
                    level_idx,
                    valid,
                ].long()
                if selected_positions.numel():
                    if (
                        (selected_positions < 0).any()
                        or (selected_positions >= query_len).any()
                    ):
                        raise RuntimeError(
                            "Teacher top-k position is outside the student query."
                        )
                    expected_ids = topk_token_ids[
                        batch_idx,
                        level_idx,
                        valid,
                    ].long()
                    actual_ids = input_ids[batch_idx, selected_positions]
                    if not torch.equal(actual_ids, expected_ids):
                        raise RuntimeError(
                            "Teacher/student tokenization mismatch at attention top-k "
                            f"sample={batch_idx}, level={level_idx}."
                        )
                    selected_scores = text_scores[selected_positions]
                    result[
                        batch_idx,
                        level_idx,
                        : selected_scores.numel(),
                    ] = selected_scores
                    result[batch_idx, level_idx, -1] = (
                        1.0 - selected_scores.sum()
                    ).clamp_min(0)
                else:
                    result[batch_idx, level_idx, -1] = 1.0
        bins = result / result.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return bins, text_mass

    def compute_loss(
        self,
        model: DistillStudent,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        labels = inputs.pop("labels")
        label_mask = inputs.pop("label_mask")
        inputs.pop("dataset_idx", None)
        teacher_global_logits = inputs.pop("teacher_global_logits", None)
        teacher_local_logits = inputs.pop("teacher_local_logits", None)
        teacher_features = inputs.pop("teacher_level_features", None)
        topk_positions = inputs.pop("topk_positions", None)
        topk_token_ids = inputs.pop("topk_token_ids", None)
        topk_valid_mask = inputs.pop("topk_valid_mask", None)
        teacher_topk = inputs.pop("teacher_topk_attention_mean", None)
        teacher_other = inputs.pop("teacher_other_attention_mean", None)
        teacher_text_mass = inputs.pop("teacher_text_attention_mass", None)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if self.attention_capture is not None:
            self.attention_capture.clear()
            self.attention_capture.enabled = True
        try:
            outputs = model(**inputs)
        finally:
            if self.attention_capture is not None:
                self.attention_capture.enabled = False
        local_logits = torch.cat(outputs["logits_list"], dim=-1).float()
        global_logits = outputs["global_logits"].float()
        flat_targets, flat_mask = build_flat_targets(
            labels,
            label_mask,
            self.level_classes,
            self.level_offsets,
        )
        flat_targets = flat_targets.to(global_logits.device)
        flat_mask = flat_mask.to(global_logits.device)

        total_loss = global_logits.sum() * 0.0
        loss_parts: dict[str, torch.Tensor] = {}

        if self.global_supervised_bce_weight > 0:
            matrix = F.binary_cross_entropy_with_logits(
                global_logits,
                flat_targets,
                reduction="none",
            )
            value = masked_mean(matrix, flat_mask)
            total_loss = (
                total_loss + self.global_supervised_bce_weight * value
            )
            loss_parts["global_supervised_bce"] = value

        if self.local_supervised_bce_weight > 0:
            matrix = F.binary_cross_entropy_with_logits(
                local_logits,
                flat_targets,
                reduction="none",
            )
            value = masked_mean(matrix, flat_mask)
            total_loss = total_loss + self.local_supervised_bce_weight * value
            loss_parts["local_supervised_bce"] = value

        if self.global_logit_kd_weight > 0:
            if teacher_global_logits is None:
                raise RuntimeError("Global-logit KD is enabled but logits are absent.")
            teacher_soft = torch.sigmoid(
                teacher_global_logits.float().to(global_logits.device)
                / self.temperature
            )
            matrix = F.binary_cross_entropy_with_logits(
                global_logits / self.temperature,
                teacher_soft,
                reduction="none",
            )
            value = masked_mean(matrix, flat_mask) * (self.temperature ** 2)
            total_loss = total_loss + self.global_logit_kd_weight * value
            loss_parts["global_logit_kd"] = value

        if self.local_logit_kd_weight > 0:
            if teacher_local_logits is None:
                raise RuntimeError("Local-logit KD is enabled but logits are absent.")
            teacher_soft = torch.sigmoid(
                teacher_local_logits.float().to(local_logits.device)
                / self.temperature
            )
            matrix = F.binary_cross_entropy_with_logits(
                local_logits / self.temperature,
                teacher_soft,
                reduction="none",
            )
            value = masked_mean(matrix, flat_mask) * (self.temperature ** 2)
            total_loss = total_loss + self.local_logit_kd_weight * value
            loss_parts["local_logit_kd"] = value

        if self.feature_kd_weight > 0:
            if teacher_features is None:
                raise RuntimeError("Feature KD is enabled but features are absent.")
            projected = outputs["projected_level_features"].float()
            teacher_features = teacher_features.float().to(projected.device)
            feature_mask = label_mask.float().to(projected.device)
            mse_per_level = (projected - teacher_features).pow(2).mean(dim=-1)
            cosine_per_level = 1.0 - F.cosine_similarity(
                projected,
                teacher_features,
                dim=-1,
            )
            mse_value = masked_mean(mse_per_level, feature_mask)
            cosine_value = masked_mean(cosine_per_level, feature_mask)
            value = mse_value + self.feature_cosine_weight * cosine_value
            total_loss = total_loss + self.feature_kd_weight * value
            loss_parts["feature_kd"] = value

        attention_enabled = (
            self.attention_kd_weight > 0
            or self.attention_mass_kd_weight > 0
        )
        if attention_enabled:
            required = [
                topk_positions,
                topk_token_ids,
                topk_valid_mask,
                teacher_topk,
                teacher_other,
                teacher_text_mass,
            ]
            if any(value is None for value in required):
                raise RuntimeError("Attention KD is enabled but attention data are absent.")
            assert self.attention_capture is not None
            student_attention = self.attention_capture.get()
            student_bins, student_text_mass = self._attention_bins(
                student_attention,
                input_ids,
                attention_mask,
                topk_positions,
                topk_token_ids,
                topk_valid_mask,
            )
            teacher_bins = torch.cat(
                [
                    teacher_topk.float().to(student_bins.device),
                    teacher_other.float().to(student_bins.device).unsqueeze(-1),
                ],
                dim=-1,
            )
            teacher_bins = teacher_bins / teacher_bins.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
            level_mask = label_mask.float().to(student_bins.device)
            if self.attention_kd_weight > 0:
                per_level = F.kl_div(
                    student_bins.clamp_min(1e-8).log(),
                    teacher_bins,
                    reduction="none",
                ).sum(dim=-1)
                value = masked_mean(per_level, level_mask)
                total_loss = total_loss + self.attention_kd_weight * value
                loss_parts["attention_kd"] = value
            if self.attention_mass_kd_weight > 0:
                teacher_mass = teacher_text_mass.float().to(
                    student_text_mass.device
                )
                mass_per_level = (student_text_mass - teacher_mass).pow(2)
                mass_value = masked_mean(mass_per_level, level_mask)
                total_loss = (
                    total_loss
                    + self.attention_mass_kd_weight * mass_value
                )
                loss_parts["attention_mass_kd"] = mass_value

        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            self._last_distill_losses = {
                key: float(value.detach().item())
                for key, value in loss_parts.items()
            }
        return (total_loss, outputs) if return_outputs else total_loss

    def log(
        self,
        logs: dict[str, float],
        start_time: float | None = None,
    ) -> None:
        if "loss" in logs and hasattr(self, "_last_distill_losses"):
            logs = {**logs, **self._last_distill_losses}
        try:
            super().log(logs, start_time=start_time)
        except TypeError:
            super().log(logs)

    def _selection_spec(
        self,
        metric_key_prefix: str,
    ) -> tuple[str, bool, str] | None:
        specs = {
            "global_macro": (
                f"{metric_key_prefix}_global_flat_macro_f1",
                True,
                "best_global_macro",
            ),
            "local_macro": (
                f"{metric_key_prefix}_local_flat_macro_f1",
                True,
                "best_local_macro",
            ),
            "global_supervised_bce": (
                f"{metric_key_prefix}_global_supervised_bce_loss",
                False,
                "best_global_supervised_bce",
            ),
            "local_supervised_bce": (
                f"{metric_key_prefix}_local_supervised_bce_loss",
                False,
                "best_local_supervised_bce",
            ),
        }
        return specs.get(self.early_stopping_target)

    def save_model(
        self,
        output_dir: str | None = None,
        _internal_call: bool = False,
    ) -> None:
        if output_dir is None:
            output_dir = self.args.output_dir
        if not self.is_world_process_zero():
            return
        os.makedirs(output_dir, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        backbone = model.core.backbone
        save_kwargs: dict[str, Any] = {
            "safe_serialization": getattr(self.args, "save_safetensors", True),
        }
        if "save_embedding_layers" in inspect.signature(
            backbone.save_pretrained
        ).parameters:
            save_kwargs["save_embedding_layers"] = False
        backbone.save_pretrained(output_dir, **save_kwargs)
        self.tokenizer_for_save.save_pretrained(output_dir)
        classifier_state = {
                "level_heads": model.core.level_heads.state_dict(),
                "global_head": model.core.global_head.state_dict(),
                "post_ln": model.core.post_ln.state_dict(),
                "num_levels": len(self.level_classes),
                "level_classes": self.level_classes,
                "level_offsets": self.level_offsets,
                "local_loss_weight": float(
                    self.local_supervised_bce_weight
                    + self.local_logit_kd_weight
                ),
                "global_loss_weight": float(
                    self.global_supervised_bce_weight
                    + self.global_logit_kd_weight
                ),
                "has_trained_local_heads": bool(
                    self.local_supervised_bce_weight > 0
                    or self.local_logit_kd_weight > 0
                ),
                "has_trained_global_head": bool(
                    self.global_supervised_bce_weight > 0
                    or self.global_logit_kd_weight > 0
                ),
        }
        torch.save(
            {
                **classifier_state,
                "feature_projection": model.feature_projection.state_dict(),
                "level_token_ids": self.level_token_ids,
                "tuning_mode": model.tuning_mode,
            },
            os.path.join(output_dir, "distill_head.pt"),
        )
        torch.save(
            classifier_state,
            os.path.join(output_dir, "classifier_head.pt"),
        )
        write_json(
            os.path.join(output_dir, "distill_config.json"),
            self.distill_config,
        )
        compatibility_config = dict(
            self.distill_config.get("evaluation_compatibility", {})
        )
        if compatibility_config:
            write_json(
                os.path.join(output_dir, "teacher_config.json"),
                compatibility_config,
            )
        write_json(
            os.path.join(output_dir, "distill_trainer_state.json"),
            {
                "early_stopping_target": self.early_stopping_target,
                "best_selection_score": self.best_selection_score,
                "best_global_micro_score": self.best_global_micro_score,
                "early_stop_bad_count": self.early_stop_bad_count,
                "global_step": int(self.state.global_step),
            },
        )

    def _load_from_checkpoint(
        self,
        resume_from_checkpoint: str,
        model: nn.Module | None = None,
    ) -> None:
        rng_name = (
            f"rng_state_{self.args.process_index}.pth"
            if self.args.world_size > 1
            else "rng_state.pth"
        )
        required_training_state = [
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            rng_name,
        ]
        if self.args.fp16:
            required_training_state.append("scaler.pt")
        missing_training_state = [
            name
            for name in required_training_state
            if not os.path.exists(os.path.join(resume_from_checkpoint, name))
        ]
        if missing_training_state:
            raise FileNotFoundError(
                "The requested directory is not a complete resumable Trainer "
                f"checkpoint; missing: {missing_training_state}"
            )

        saved_config_path = os.path.join(
            resume_from_checkpoint,
            "distill_config.json",
        )
        if not os.path.exists(saved_config_path):
            raise FileNotFoundError(
                f"Checkpoint lacks distill_config.json: {resume_from_checkpoint}"
            )
        saved_config = read_json(saved_config_path)
        compatibility_keys = (
            "dataset",
            "student_model_path",
            "teacher_checkpoint_dir",
            "knowledge_dir",
            "tuning_mode",
            "num_levels",
            "level_classes",
            "level_offsets",
            "student_level_token_ids",
            "teacher_level_token_ids",
            "loss_weights",
            "temperature",
            "student_feature_layer",
            "student_attention_layer",
            "early_stopping_target",
            "seed",
            "training_compatibility",
        )
        mismatches = [
            key
            for key in compatibility_keys
            if saved_config.get(key) != self.distill_config.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "Resume checkpoint configuration differs from the current run "
                f"for: {mismatches}"
            )

        target = model if model is not None else self.model
        target = self.accelerator.unwrap_model(target)
        if not isinstance(target, DistillStudent):
            raise TypeError(
                "Checkpoint restoration expected an unwrapped DistillStudent, "
                f"got {type(target).__name__}."
            )

        backbone = target.core.backbone
        if target.tuning_mode == "lora":
            adapter_path = os.path.join(
                resume_from_checkpoint,
                "adapter_model.safetensors",
            )
            if os.path.exists(adapter_path):
                adapter_state = load_file(adapter_path, device="cpu")
            else:
                adapter_path = os.path.join(
                    resume_from_checkpoint,
                    "adapter_model.bin",
                )
                if not os.path.exists(adapter_path):
                    raise FileNotFoundError(
                        "LoRA checkpoint lacks adapter_model.safetensors or "
                        f"adapter_model.bin: {resume_from_checkpoint}"
                    )
                adapter_state = load_trusted_torch_file(adapter_path)
            set_peft_model_state_dict(
                backbone,
                adapter_state,
                adapter_name="default",
            )
        else:
            safe_index = os.path.join(
                resume_from_checkpoint,
                "model.safetensors.index.json",
            )
            bin_index = os.path.join(
                resume_from_checkpoint,
                "pytorch_model.bin.index.json",
            )
            if os.path.exists(safe_index) or os.path.exists(bin_index):
                load_sharded_backbone_checkpoint(
                    backbone,
                    resume_from_checkpoint,
                    prefer_safe=os.path.exists(safe_index),
                )
            else:
                safe_path = os.path.join(
                    resume_from_checkpoint,
                    "model.safetensors",
                )
                bin_path = os.path.join(
                    resume_from_checkpoint,
                    "pytorch_model.bin",
                )
                if os.path.exists(safe_path):
                    backbone_state = load_file(safe_path, device="cpu")
                elif os.path.exists(bin_path):
                    backbone_state = load_trusted_torch_file(bin_path)
                else:
                    raise FileNotFoundError(
                        "Full-finetuning checkpoint lacks model weights: "
                        f"{resume_from_checkpoint}"
                    )
                backbone.load_state_dict(backbone_state, strict=True)

        head_path = os.path.join(resume_from_checkpoint, "distill_head.pt")
        if not os.path.exists(head_path):
            raise FileNotFoundError(
                f"Checkpoint lacks distill_head.pt: {resume_from_checkpoint}"
            )
        head_state = load_trusted_torch_file(head_path)
        target.core.level_heads.load_state_dict(head_state["level_heads"])
        target.core.global_head.load_state_dict(head_state["global_head"])
        target.core.post_ln.load_state_dict(head_state["post_ln"])
        target.feature_projection.load_state_dict(
            head_state["feature_projection"]
        )
        if list(head_state["level_token_ids"]) != self.level_token_ids:
            raise RuntimeError(
                "Checkpoint level token ids differ from the current tokenizer."
            )
        if head_state.get("tuning_mode") != target.tuning_mode:
            raise RuntimeError(
                "Checkpoint tuning mode differs from the requested tuning mode."
            )

        custom_state_path = os.path.join(
            resume_from_checkpoint,
            "distill_trainer_state.json",
        )
        if not os.path.exists(custom_state_path):
            raise FileNotFoundError(
                "Checkpoint lacks distill_trainer_state.json and cannot restore "
                "custom early-stopping state."
            )
        custom_state = read_json(custom_state_path)
        if (
            custom_state.get("early_stopping_target")
            != self.early_stopping_target
        ):
            raise RuntimeError(
                "Checkpoint early-stopping target differs from the current "
                "configuration."
            )
        value = custom_state.get("best_selection_score")
        self.best_selection_score = None if value is None else float(value)
        micro_value = custom_state.get("best_global_micro_score")
        self.best_global_micro_score = (
            None if micro_value is None else float(micro_value)
        )
        self.early_stop_bad_count = int(
            custom_state.get("early_stop_bad_count", 0)
        )

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model_wrapped
        model.eval()
        all_logits: list[np.ndarray] = []
        all_local_logits: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []
        global_supervised_bce_sum = 0.0
        local_supervised_bce_sum = 0.0
        loss_count = 0
        for batch in dataloader:
            batch = self._prepare_inputs(batch)
            labels = batch.pop("labels")
            label_mask = batch.pop("label_mask")
            batch.pop("dataset_idx", None)
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs["global_logits"].float()
                local_logits = torch.cat(
                    outputs["logits_list"],
                    dim=-1,
                ).float()
                targets, mask = build_flat_targets(
                    labels,
                    label_mask,
                    self.level_classes,
                    self.level_offsets,
                )
                (
                    gathered_logits,
                    gathered_local_logits,
                    gathered_targets,
                    gathered_mask,
                ) = (
                    self.accelerator.gather_for_metrics(
                        (
                            logits.detach(),
                            local_logits.detach(),
                            targets.detach().float(),
                            mask.detach().float(),
                        )
                    )
                )
                gathered_matrix = F.binary_cross_entropy_with_logits(
                    gathered_logits,
                    gathered_targets,
                    reduction="none",
                )
                gathered_local_matrix = F.binary_cross_entropy_with_logits(
                    gathered_local_logits,
                    gathered_targets,
                    reduction="none",
                )
                global_supervised_bce_sum += (
                    gathered_matrix * gathered_mask
                ).sum().detach().item()
                local_supervised_bce_sum += (
                    gathered_local_matrix * gathered_mask
                ).sum().detach().item()
                loss_count += int(gathered_mask.sum().detach().item())
                all_logits.append(gathered_logits.cpu().numpy())
                all_local_logits.append(gathered_local_logits.cpu().numpy())
                all_targets.append(gathered_targets.cpu().numpy())
                all_masks.append(gathered_mask.cpu().numpy())

        logits_np = np.concatenate(all_logits, axis=0)
        local_logits_np = np.concatenate(all_local_logits, axis=0)
        targets_np = np.concatenate(all_targets, axis=0).astype(np.int32)
        masks_np = np.concatenate(all_masks, axis=0).astype(bool)
        predictions = (logits_np > self.fixed_threshold).astype(np.int32)
        local_predictions = (
            local_logits_np > self.fixed_threshold
        ).astype(np.int32)
        metrics = {
            f"{metric_key_prefix}_global_supervised_bce_loss": (
                global_supervised_bce_sum / max(1, loss_count)
            ),
            f"{metric_key_prefix}_local_supervised_bce_loss": (
                local_supervised_bce_sum / max(1, loss_count)
            ),
            f"{metric_key_prefix}_global_flat_macro_f1": masked_multilabel_f1(
                targets_np,
                predictions,
                masks_np,
                average="macro",
            ),
            f"{metric_key_prefix}_global_flat_micro_f1": masked_multilabel_f1(
                targets_np,
                predictions,
                masks_np,
                average="micro",
            ),
        }
        if (
            self.local_supervised_bce_weight > 0
            or self.local_logit_kd_weight > 0
        ):
            metrics[f"{metric_key_prefix}_local_flat_macro_f1"] = (
                masked_multilabel_f1(
                    targets_np,
                    local_predictions,
                    masks_np,
                    average="macro",
                )
            )
            metrics[f"{metric_key_prefix}_local_flat_micro_f1"] = (
                masked_multilabel_f1(
                    targets_np,
                    local_predictions,
                    masks_np,
                    average="micro",
                )
            )
        for level_idx, width in enumerate(self.level_classes):
            start = self.level_offsets[level_idx]
            end = start + width
            valid_rows = masks_np[:, start:end].any(axis=1)
            if not valid_rows.any():
                continue
            level_targets = targets_np[valid_rows, start:end]
            level_global_predictions = predictions[valid_rows, start:end]
            level_id = level_idx + 1
            metrics[
                f"{metric_key_prefix}_global_level{level_id}_macro_f1"
            ] = float(
                f1_score(
                    level_targets,
                    level_global_predictions,
                    average="macro",
                    zero_division=0,
                )
            )
            metrics[
                f"{metric_key_prefix}_global_level{level_id}_micro_f1"
            ] = float(
                f1_score(
                    level_targets,
                    level_global_predictions,
                    average="micro",
                    zero_division=0,
                )
            )
            if (
                self.local_supervised_bce_weight > 0
                or self.local_logit_kd_weight > 0
            ):
                level_local_predictions = local_predictions[
                    valid_rows,
                    start:end,
                ]
                metrics[
                    f"{metric_key_prefix}_local_level{level_id}_macro_f1"
                ] = float(
                    f1_score(
                        level_targets,
                        level_local_predictions,
                        average="macro",
                        zero_division=0,
                    )
                )
                metrics[
                    f"{metric_key_prefix}_local_level{level_id}_micro_f1"
                ] = float(
                    f1_score(
                        level_targets,
                        level_local_predictions,
                        average="micro",
                        zero_division=0,
                    )
                )
        self.log(metrics)

        if metric_key_prefix == "eval":
            selection = self._selection_spec(metric_key_prefix)
            if selection is not None:
                metric_key, greater_is_better, directory_name = selection
                current = float(metrics[metric_key])
                improved = (
                    self.best_selection_score is None
                    or (
                        current > self.best_selection_score
                        if greater_is_better
                        else current < self.best_selection_score
                    )
                )
            else:
                metric_key = ""
                directory_name = ""
                current = 0.0
                improved = False
            if selection is not None and improved:
                previous_best = self.best_selection_score
                self.best_selection_score = current
                self.early_stop_bad_count = 0
                if self.is_world_process_zero():
                    print(
                        f"[EarlyStopping] step={int(self.state.global_step)} | "
                        f"metric={metric_key} | "
                        f"current={current:.6f} | "
                        f"previous_best="
                        f"{'None' if previous_best is None else f'{previous_best:.6f}'} | "
                        f"NEW BEST | "
                        f"patience=0/{self.early_stopping_patience}"
                    )
                self.save_model(
                    os.path.join(self.args.output_dir, directory_name)
                )
                if self.is_world_process_zero():
                    write_json(
                        os.path.join(
                            self.args.output_dir,
                            directory_name,
                            "best_metrics.json",
                        ),
                        {
                            "selected_by": metric_key,
                            "selected_value": current,
                            "global_step": int(self.state.global_step),
                            "metrics": metrics,
                        },
                    )
            elif selection is not None:
                self.early_stop_bad_count += 1

                if self.is_world_process_zero():
                    print(
                        f"[EarlyStopping] step={int(self.state.global_step)} | "
                        f"metric={metric_key} | "
                        f"current={current:.6f} | "
                        f"best={self.best_selection_score:.6f} | "
                        f"NO IMPROVEMENT | "
                        f"patience={self.early_stop_bad_count}/"
                        f"{self.early_stopping_patience}"
                    )

                if (
                    self.early_stopping_patience > 0
                    and self.early_stop_bad_count
                    >= self.early_stopping_patience
                ):
                    if self.is_world_process_zero():
                        print(
                            f"[EarlyStopping] TRIGGERED at "
                            f"step={int(self.state.global_step)} | "
                            f"best={self.best_selection_score:.6f} | "
                            f"patience={self.early_stop_bad_count}/"
                            f"{self.early_stopping_patience}"
                        )
                    self.control.should_training_stop = True

            # ---------------------------------------------------------
            # Independently save best GLOBAL MICRO F1.
            # This does NOT participate in early stopping.
            # ---------------------------------------------------------
            micro_key = f"{metric_key_prefix}_global_flat_micro_f1"

            global_head_trained = (
                self.global_supervised_bce_weight > 0
                or self.global_logit_kd_weight > 0
            )

            if global_head_trained and micro_key in metrics:
                current_micro = float(metrics[micro_key])

                micro_improved = (
                    self.best_global_micro_score is None
                    or current_micro > self.best_global_micro_score
                )

                if micro_improved:
                    previous_micro = self.best_global_micro_score
                    self.best_global_micro_score = current_micro

                    if self.is_world_process_zero():
                        print(
                            f"[BestMicro] step={int(self.state.global_step)} | "
                            f"metric={micro_key} | "
                            f"current={current_micro:.6f} | "
                            f"previous_best="
                            f"{'None' if previous_micro is None else f'{previous_micro:.6f}'} | "
                            f"NEW BEST"
                        )

                    micro_dir = os.path.join(
                        self.args.output_dir,
                        "best_global_micro",
                    )

                    self.save_model(micro_dir)

                    if self.is_world_process_zero():
                        write_json(
                            os.path.join(
                                micro_dir,
                                "best_metrics.json",
                            ),
                            {
                                "selected_by": micro_key,
                                "selected_value": current_micro,
                                "global_step": int(self.state.global_step),
                                "metrics": metrics,
                            },
                        )
            self.control = self.callback_handler.on_evaluate(
                self.args,
                self.state,
                self.control,
                metrics,
            )
        model.train()
        return metrics


def load_student_backbone(
    model_path: str,
    tokenizer: Any,
    args: argparse.Namespace,
    need_attention: bool,
) -> nn.Module:
    can_use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = (
        torch.bfloat16
        if can_use_bf16
        else torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if need_attention:
        load_kwargs["attn_implementation"] = "eager"
    try:
        backbone = AutoModel.from_pretrained(model_path, **load_kwargs)
    except (TypeError, ValueError) as exc:
        if not is_attention_implementation_compat_error(exc):
            raise
        load_kwargs.pop("attn_implementation", None)
        backbone = AutoModel.from_pretrained(model_path, **load_kwargs)
        if need_attention and hasattr(backbone, "set_attn_implementation"):
            backbone.set_attn_implementation("eager")
    backbone.config.use_cache = False
    if len(tokenizer) > backbone.get_input_embeddings().num_embeddings:
        backbone.resize_token_embeddings(len(tokenizer))

    if args.tuning_mode == "full":
        return backbone
    if "trainable_token_indices" not in inspect.signature(
        LoraConfig.__init__
    ).parameters:
        raise RuntimeError(
            "This PEFT version lacks trainable_token_indices support."
        )
    level_token_ids = tokenizer.convert_tokens_to_ids(
        LEVEL_TOKENS[: args.num_levels]
    )
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        trainable_token_indices={"embed_tokens": level_token_ids},
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()
    return backbone


def main() -> None:
    args = parse_args()
    validate_loss_args(args)
    seed_everything(args.seed)
    attention_enabled = (
        args.attention_kd_weight > 0
        or args.attention_mass_kd_weight > 0
    )
    if attention_enabled and not args.attention_cache_dir:
        raise ValueError("Attention KD requires --attention_cache_dir with the compact FP32 bank")
    if args.attention_cache_samples < 0:
        raise ValueError("--attention_cache_samples must be nonnegative")
    student_attention_layer = (
        args.student_feature_layer
        if args.student_attention_layer is None
        else args.student_attention_layer
    )

    knowledge_dir = os.path.abspath(args.knowledge_dir)
    manifest_path = os.path.join(knowledge_dir, "manifest.json")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError(
            f"Knowledge manifest is not complete: status={manifest.get('status')}"
        )
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(
            "Knowledge schema is incompatible with this trainer. Re-extract or "
            "complete the knowledge with the current extractor."
        )
    if manifest.get("dataset") != args.dataset or manifest.get("split") != "train":
        raise RuntimeError("Knowledge dataset/split differs from requested training data.")
    # Attention replay validation and teacher-layer checks use the compact
    # bank below. The original manifest supplies only logits/features here.
    if (
        args.global_logit_kd_weight > 0
        and not bool(manifest.get("has_trained_global_head", False))
    ):
        raise RuntimeError("Global-logit KD requires a trained teacher global head.")
    if (
        args.local_logit_kd_weight > 0
        and not bool(manifest.get("has_trained_local_heads", False))
    ):
        raise RuntimeError("Local-logit KD requires trained teacher local heads.")

    dataset_cfg = DATASET_ZOO[args.dataset]
    data_dir = args.data_dir or dataset_cfg["path"]
    train_path = os.path.join(data_dir, dataset_cfg["train_file"])
    dev_path = os.path.join(data_dir, dataset_cfg["dev_file"])
    if sha256_file(train_path) != manifest["data_sha256"]:
        raise RuntimeError("Current train JSONL differs from the extracted knowledge source.")

    train_meta = load_preprocess_meta(train_path)
    dev_meta = load_preprocess_meta(dev_path)
    assert_preprocess_meta_compatible(train_meta, dev_meta, "dev")
    num_levels = train_meta["num_levels"]
    level_classes = train_meta["level_classes"]
    total_classes = train_meta["total_classes"]
    level_offsets = build_level_offsets(level_classes)
    if list(manifest["level_classes"]) != level_classes:
        raise RuntimeError("Knowledge level_classes differ from current metadata.")
    if list(manifest["level_offsets"]) != level_offsets:
        raise RuntimeError("Knowledge level_offsets differ from current metadata.")

    required_modules: set[str] = set()
    if args.global_logit_kd_weight > 0 or args.local_logit_kd_weight > 0:
        required_modules.add("logits")
    if args.feature_kd_weight > 0:
        required_modules.add("features")

    identities = build_sample_identities(
        train_path,
        args.dataset,
        "train",
        num_levels,
    )
    expected_uids = [item["sample_uid"] for item in identities]
    knowledge = OfflineKnowledge(
        knowledge_dir,
        manifest,
        required_modules,
        expected_uids,
    )
    attention_knowledge = None
    if attention_enabled:
        attention_knowledge = CachedLevelTextKnowledge(
            args.attention_cache_dir, manifest, identities,
            args.attention_topk, args.attention_cache_samples,
        )
        knowledge = CombinedCachedKnowledge(knowledge, attention_knowledge)

    model_path = args.model_path or MODEL_ZOO[args.model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base_tokenizer_vocab_size = len(tokenizer)
    num_added_level_tokens = add_level_special_tokens(
        tokenizer,
        LEVEL_TOKENS[:num_levels],
    )
    student_level_token_ids = tokenizer.convert_tokens_to_ids(
        LEVEL_TOKENS[:num_levels]
    )
    if attention_enabled:
        student_tokenizer_hash = tokenizer_fingerprint(tokenizer)
        if student_tokenizer_hash != manifest["tokenizer_sha256"]:
            raise RuntimeError(
                "Teacher and student tokenizer vocabularies differ. "
                "Attention KD requires exact tokenizer alignment in this version."
            )

    max_length = int(manifest["max_length"])
    truncate_mode = str(manifest["truncate_mode"])
    train_base = HierarchicalDataset(
        train_path,
        tokenizer,
        num_levels,
        total_classes,
        level_offsets,
        max_length,
        truncate_mode,
        strict=True,
    )
    dev_dataset = HierarchicalDataset(
        dev_path,
        tokenizer,
        num_levels,
        total_classes,
        level_offsets,
        max_length,
        truncate_mode,
        strict=True,
    )
    train_dataset = DistillTrainDataset(train_base, knowledge)

    args.num_levels = num_levels
    backbone = load_student_backbone(
        model_path,
        tokenizer,
        args,
        need_attention=attention_enabled,
    )
    num_student_layers = int(backbone.config.num_hidden_layers)
    if not (
        -num_student_layers
        <= args.student_feature_layer
        < num_student_layers
    ):
        raise ValueError(
            f"--student_feature_layer={args.student_feature_layer} is outside "
            f"[-{num_student_layers}, {num_student_layers})."
        )
    core = HierarchicalTeacher(
        backbone=backbone,
        tokenizer=tokenizer,
        num_levels=num_levels,
        level_classes=level_classes,
        level_offsets=level_offsets,
        hidden_layer_idx=args.student_feature_layer,
        use_hidden_states=args.student_feature_layer != -1,
        dropout=args.dropout,
    )
    model = DistillStudent(
        core,
        teacher_hidden_size=int(manifest["teacher_hidden_size"]),
        tuning_mode=args.tuning_mode,
        enable_feature_projection=args.feature_kd_weight > 0,
    )
    if (
        args.local_supervised_bce_weight == 0
        and args.local_logit_kd_weight == 0
    ):
        model.core.level_heads.requires_grad_(False)
    if (
        args.global_supervised_bce_weight == 0
        and args.global_logit_kd_weight == 0
    ):
        model.core.global_head.requires_grad_(False)
    if args.feature_kd_weight == 0:
        model.feature_projection.requires_grad_(False)
    if (
        args.global_supervised_bce_weight == 0
        and args.local_supervised_bce_weight == 0
        and args.global_logit_kd_weight == 0
        and args.local_logit_kd_weight == 0
        and args.feature_kd_weight == 0
    ):
        model.core.post_ln.requires_grad_(False)

    output_dir = args.output_dir or (
        f"/data1/lwq/agt_plus/output_distill/"
        f"{args.dataset}_{Path(model_path).name}_distill"
    )
    run_name = args.run_name or Path(output_dir).name
    os.makedirs(output_dir, exist_ok=True)
    # Save the exact command used for this run
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        command_parts = []

        for env_name in (
                "CUDA_VISIBLE_DEVICES",
                "SWANLAB_PROJECT",
                "PYTORCH_CUDA_ALLOC_CONF",
        ):
            env_value = os.environ.get(env_name)
            if env_value is not None:
                command_parts.append(f"{env_name}={env_value}")

        command_parts.extend(
            [sys.executable] + [str(arg) for arg in sys.argv]
        )

        with open(
                os.path.join(output_dir, "command.txt"),
                "w",
                encoding="utf-8",
        ) as f:
            f.write(" ".join(command_parts) + "\n")

        write_json(
            os.path.join(output_dir, "args.json"),
            vars(args),
        )
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        sys.stdout = TeeLogger(os.path.join(output_dir, "training.log"))
        sys.stderr = sys.stdout

    selection_specs = {
        "global_macro": ("eval_global_flat_macro_f1", True),
        "local_macro": ("eval_local_flat_macro_f1", True),
        "global_supervised_bce": (
            "eval_global_supervised_bce_loss",
            False,
        ),
        "local_supervised_bce": (
            "eval_local_supervised_bce_loss",
            False,
        ),
    }
    selection_spec = selection_specs.get(args.early_stopping_target)
    selection_metric = selection_spec[0] if selection_spec is not None else None
    selection_greater_is_better = (
        selection_spec[1] if selection_spec is not None else None
    )
    can_use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        data_seed=args.seed,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        bf16=can_use_bf16,
        fp16=not can_use_bf16 and torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        report_to=[] if args.no_swanlab else ["swanlab"],
        run_name=run_name,
        dataloader_pin_memory=True,
        dataloader_num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        metric_for_best_model=selection_metric,
        greater_is_better=selection_greater_is_better,
        load_best_model_at_end=False,
    )

    distill_config = {
        "dataset": args.dataset,
        "student_model_path": model_path,
        "teacher_checkpoint_dir": manifest["checkpoint_dir"],
        "knowledge_dir": knowledge_dir,
        "tuning_mode": args.tuning_mode,
        "num_levels": num_levels,
        "level_classes": level_classes,
        "level_offsets": level_offsets,
        "level_tokens": LEVEL_TOKENS[:num_levels],
        "student_level_token_ids": student_level_token_ids,
        "teacher_level_token_ids": manifest["teacher_level_token_ids"],
        "max_length": max_length,
        "truncate_mode": truncate_mode,
        "feature_type": manifest["feature_type"],
        "attention_topk": attention_knowledge.topk if attention_knowledge else manifest["attention_topk"],
        "fixed_threshold": args.fixed_threshold,
        "loss_weights": {
            "global_supervised_bce": args.global_supervised_bce_weight,
            "local_supervised_bce": args.local_supervised_bce_weight,
            "global_logit_kd": args.global_logit_kd_weight,
            "local_logit_kd": args.local_logit_kd_weight,
            "feature_kd": args.feature_kd_weight,
            "feature_cosine": args.feature_cosine_weight,
            "attention_kd": args.attention_kd_weight,
            "attention_mass_kd": args.attention_mass_kd_weight,
        },
        "student_feature_layer": args.student_feature_layer,
        "student_attention_layer": student_attention_layer,
        "teacher_feature_layer": manifest.get("feature_layer"),
        "teacher_attention_layer": attention_knowledge.layer if attention_knowledge else manifest.get("attention_layer"),
        "feature_kd_alignment": "post_ln_linear_projection",
        "student_attention_capture": "same_forward_selected_layer",
        "attention_mass_loss": "masked_mse",
        "temperature": args.temperature,
        "seed": args.seed,
        "training_compatibility": {
            **({"cached_attention": attention_knowledge.compatibility()} if attention_knowledge else {}),
            "train_batch": args.train_batch,
            "gradient_accumulation_steps": args.grad_acc,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "early_stopping_target": args.early_stopping_target,
        "early_stopping_metric": selection_metric,
    }
    distill_config["evaluation_compatibility"] = {
        "dataset": args.dataset,
        "model_path": model_path,
        "run_name": run_name,
        "num_levels": num_levels,
        "level_classes": level_classes,
        "level_offsets": level_offsets,
        "label_space_source": train_meta["meta"].get("label_space_source"),
        "layer_label2id": normalize_layer_label2id(train_meta["meta"]),
        "level_tokens": LEVEL_TOKENS[:num_levels],
        "level_token_ids": student_level_token_ids,
        "base_tokenizer_vocab_size": base_tokenizer_vocab_size,
        "tokenizer_vocab_size": len(tokenizer),
        "num_added_level_tokens": num_added_level_tokens,
        "level_tokens_are_new": bool(
            num_added_level_tokens == num_levels
            and all(
                token_id >= base_tokenizer_vocab_size
                for token_id in student_level_token_ids
            )
        ),
        "embedding_training": (
            "trainable_token_indices"
            if args.tuning_mode == "lora"
            else "full_finetuning"
        ),
        "hidden_layer_idx": args.student_feature_layer,
        "hidden_layer_indexing": "transformer_block_0_based",
        "hidden_states_index": (
            args.student_feature_layer
            if args.student_feature_layer < 0
            else args.student_feature_layer + 1
        ),
        "use_hidden_states": args.student_feature_layer != -1,
        "max_length": max_length,
        "truncate_mode": truncate_mode,
        "level_loss_weights": [1.0] * num_levels,
        "local_loss_weight": float(
            args.local_supervised_bce_weight + args.local_logit_kd_weight
        ),
        "global_loss_weight": float(
            args.global_supervised_bce_weight + args.global_logit_kd_weight
        ),
        "early_stop_target": args.early_stopping_target,
        "early_stop_metric": selection_metric,
        "fixed_threshold": args.fixed_threshold,
        "seed": args.seed,
        "data_seed": args.seed,
        "has_trained_local_heads": bool(
            args.local_supervised_bce_weight > 0
            or args.local_logit_kd_weight > 0
        ),
        "has_trained_global_head": bool(
            args.global_supervised_bce_weight > 0
            or args.global_logit_kd_weight > 0
        ),
        "checkpoint_format": args.tuning_mode,
    }

    trainer = DistillationTrainer(
        tokenizer_for_save=tokenizer,
        distill_config=distill_config,
        level_token_ids=student_level_token_ids,
        level_classes=level_classes,
        level_offsets=level_offsets,
        global_supervised_bce_weight=args.global_supervised_bce_weight,
        local_supervised_bce_weight=args.local_supervised_bce_weight,
        global_logit_kd_weight=args.global_logit_kd_weight,
        local_logit_kd_weight=args.local_logit_kd_weight,
        feature_kd_weight=args.feature_kd_weight,
        feature_cosine_weight=args.feature_cosine_weight,
        attention_kd_weight=args.attention_kd_weight,
        attention_mass_kd_weight=args.attention_mass_kd_weight,
        student_attention_layer=student_attention_layer,
        temperature=args.temperature,
        fixed_threshold=args.fixed_threshold,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_target=args.early_stopping_target,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=cached_topk_collator,
    )
    if trainer.attention_capture is not None:
        validation_sample = train_base[0]
        validation_device = next(trainer.model.parameters()).device
        attention_validation = validate_same_forward_attention_capture(
            trainer.model,
            trainer.attention_capture,
            validation_sample["input_ids"].unsqueeze(0).to(validation_device),
            validation_sample["attention_mask"].unsqueeze(0).to(
                validation_device
            ),
            num_levels,
        )
        if trainer.is_world_process_zero():
            print(
                "Student same-forward attention capture validation passed: "
                f"seq_len={int(attention_validation['validation_seq_len'])}, "
                f"max_row_sum_error="
                f"{attention_validation['max_row_sum_error']:.8f}, "
                f"mean_row_sum_error="
                f"{attention_validation['mean_row_sum_error']:.8f}"
            )
    resume_from_checkpoint: str | None = None
    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint.lower() == "latest":
            resume_from_checkpoint = get_last_checkpoint(output_dir)
            if resume_from_checkpoint is None:
                raise FileNotFoundError(
                    f"No checkpoint-* directory found under {output_dir}."
                )
        else:
            resume_from_checkpoint = os.path.abspath(
                args.resume_from_checkpoint
            )
            if not os.path.isdir(resume_from_checkpoint):
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {resume_from_checkpoint}"
                )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(os.path.join(output_dir, "final_model"))
    final_metrics = trainer.evaluate(
        eval_dataset=dev_dataset,
        metric_key_prefix="final_eval",
    )
    if trainer.is_world_process_zero():
        write_json(
            os.path.join(output_dir, "final_eval_metrics.json"),
            final_metrics,
        )
    if trainer.attention_capture is not None:
        trainer.attention_capture.close()


if __name__ == "__main__":
    main()
