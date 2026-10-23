"""

"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from tqdm.auto import tqdm

LEVEL_TOKENS = [
    "<|L1_CLS|>",
    "<|L2_CLS|>",
    "<|L3_CLS|>",
    "<|L4_CLS|>",
    "<|L5_CLS|>",
    "<|L6_CLS|>",
    "<|L7_CLS|>",
    "<|L8_CLS|>",
]


MODEL_ZOO = {
    "llama1b": "/data1/lwq/model/Llama-3.2-1B-Instruct",
    "llama8b": "/data1/lwq/model/llama-8b",
    "qwen0.5b": "/data1/lwq/model/Qwen2.5-0.5B",
    "qwen7b": "/data1/lwq/model/Qwen2.5-7B",
    "qwen0.6b": "/data1/lwq/model/Qwen3-0.6B-Base",
    "qwen8b": "/data1/lwq/model/Qwen3-8B-Base",
    "qwen14b": "/data1/lwq/model/Qwen2.5-14B-Instruct",
}


DATASET_ZOO = {
    "wos": {
        "path": "/data1/lwq/agt_plus/data/wos",
        "max_length": 2048,
        "train_file": "wos_train_trainer_multi.json",
        "dev_file": "wos_dev_trainer_multi.json",
        "test_file": "wos_test_trainer_multi.json",
    },
    "rcv1": {
        "path": "/data1/lwq/agt_plus/data/rcv1",
        "max_length": 2048,
        "train_file": "rcv1_train_trainer_multi.json",
        "dev_file": "rcv1_dev_trainer_multi.json",
        "test_file": "rcv1_test_trainer_multi.json",
    },
    "nyt": {
        "path": "/data1/lwq/agt_plus/data/nyt",
        "max_length": 2048,
        "train_file": "nyt_train_trainer_multi.json",
        "dev_file": "nyt_dev_trainer_multi.json",
        "test_file": "nyt_test_trainer_multi.json",
    },
    "bgc": {
        "path": "/data1/lwq/agt_plus/data/bgc",
        "max_length": 2048,
        "train_file": "bgc_train_trainer_multi.json",
        "dev_file": "bgc_dev_trainer_multi.json",
        "test_file": "bgc_test_trainer_multi.json",
    },
}
# DATASET_ZOO = {
#     "wos": {
#         "path": "/home/lwq/dataspace/data/wos",
#         "max_length": 2048,
#         "train_file": "wos_train_trainer_multi.json",
#         "dev_file": "wos_dev_trainer_multi.json",
#         "test_file": "wos_test_trainer_multi.json",
#     },
#     "rcv1": {
#         "path": "/home/lwq/dataspace/data/rcv1",
#         "max_length": 2048,
#         "train_file": "rcv1_train_trainer_multi.json",
#         "dev_file": "rcv1_dev_trainer_multi.json",
#         "test_file": "rcv1_test_trainer_multi.json",
#     },
#     "nyt": {
#         "path": "/home/lwq/dataspace/data/nyt",
#         "max_length": 4096,
#         "train_file": "nyt_train_trainer_multi.json",
#         "dev_file": "nyt_dev_trainer_multi.json",
#         "test_file": "nyt_test_trainer_multi.json",
#     },
# }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hierarchical teacher with level special tokens.")

    parser.add_argument("--dataset", default="nyt", choices=sorted(DATASET_ZOO))
    parser.add_argument("--model_key", default="qwen8b", choices=sorted(MODEL_ZOO))
    parser.add_argument("--model_path", default=None, help="Override MODEL_ZOO path.")
    parser.add_argument("--data_dir", default=None, help="Override DATASET_ZOO path.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_name", default=None, help="SwanLab/HuggingFace run name. Defaults to output_dir name.")

    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--truncate_mode", choices=["head", "head_tail"], default="head")

    parser.add_argument("--train_batch", type=int, default=8)
    parser.add_argument("--eval_batch", type=int, default=8)
    parser.add_argument("--grad_acc", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=15)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument(
        "--early_stop_target",
        choices=["local", "global"],
        default="local",
        help="Metric family for early stopping: local uses eval_flat_macro_f1, global uses eval_global_flat_macro_f1.",
    )

    parser.add_argument("--global_loss_weight", type=float, default=1.0)
    parser.add_argument("--local_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--level_loss_weights",
        default=None,
        help="Comma-separated level weights, e.g. 1,1.2,1.5. Defaults to all ones.",
    )

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--student_layer_idx", type=int, default=-1)
    parser.add_argument("--use_hidden_states", action="store_true", help="Store all hidden states and select --student_layer_idx. More memory hungry.")

    parser.add_argument("--threshold_min", type=float, default=-2.0)
    parser.add_argument("--threshold_max", type=float, default=1.0)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--fixed_threshold", type=float, default=0.0, help="Logit threshold. 0 equals sigmoid 0.5.")

    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_swanlab", action="store_true")

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def load_preprocess_meta(data_path: str) -> dict[str, Any]:
    meta_path = data_path.replace(".json", ".meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta file: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    layer_label2id = meta["layer_label2id"]
    level_classes = [len(layer) for layer in layer_label2id]
    return {
        "meta_path": meta_path,
        "num_levels": int(meta["num_levels"]),
        "level_classes": level_classes,
        "total_classes": int(sum(level_classes)),
        "meta": meta,
    }

def normalize_layer_label2id(meta: dict[str, Any]) -> list[dict[str, int]]:
    return [
        {str(name): int(label_id) for name, label_id in layer.items()}
        for layer in meta["layer_label2id"]
    ]


def assert_preprocess_meta_compatible(reference: dict[str, Any], candidate: dict[str, Any], candidate_name: str) -> None:
    ref_meta = reference["meta"]
    cand_meta = candidate["meta"]
    checks = {
        "num_levels": (reference["num_levels"], candidate["num_levels"]),
        "level_classes": (reference["level_classes"], candidate["level_classes"]),
        "label_space_source": (ref_meta.get("label_space_source"), cand_meta.get("label_space_source")),
    }
    mismatches = [
        f"{key}: reference={ref_value}, {candidate_name}={cand_value}"
        for key, (ref_value, cand_value) in checks.items()
        if ref_value != cand_value
    ]
    if normalize_layer_label2id(ref_meta) != normalize_layer_label2id(cand_meta):
        mismatches.append(f"layer_label2id differs between reference and {candidate_name}")
    if mismatches:
        raise ValueError(
            f"Preprocess meta mismatch for {candidate_name}. Re-run preprocessing with the same label space.\n"
            + "\n".join(mismatches)
        )

def build_level_offsets(level_classes: list[int]) -> list[int]:
    offsets = [0]
    for cls_num in level_classes[:-1]:
        offsets.append(offsets[-1] + cls_num)
    return offsets

def add_level_special_tokens(tokenizer: Any, level_tokens: list[str]) -> int:
    special_tokens = {"additional_special_tokens": level_tokens}
    signature = inspect.signature(tokenizer.add_special_tokens)
    if "replace_additional_special_tokens" in signature.parameters:
        return tokenizer.add_special_tokens(
            special_tokens,
            replace_additional_special_tokens=False,
        )

    existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    merged = existing + [token for token in level_tokens if token not in existing]
    return tokenizer.add_special_tokens({"additional_special_tokens": merged})

def parse_level_loss_weights(raw: str | None, num_levels: int) -> list[float]:
    if raw is None:
        return [1.0] * num_levels
    weights = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(weights) != num_levels:
        raise ValueError(f"--level_loss_weights needs {num_levels} values, got {len(weights)}")
    return weights


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def masked_multilabel_f1(trues: np.ndarray, preds: np.ndarray, mask: np.ndarray, average: str) -> float:
    valid_mask = mask.astype(bool)
    if average == "micro":
        if valid_mask.sum() == 0:
            return 0.0
        return float(f1_score(trues[valid_mask], preds[valid_mask], average="binary", zero_division=0))
    if average != "macro":
        raise ValueError(f"Unsupported average={average}")

    class_scores = []
    for class_idx in range(trues.shape[1]):
        class_mask = valid_mask[:, class_idx]
        if class_mask.sum() == 0:
            continue
        class_scores.append(
            f1_score(
                trues[class_mask, class_idx],
                preds[class_mask, class_idx],
                average="binary",
                zero_division=0,
            )
        )
    return float(np.mean(class_scores)) if class_scores else 0.0

class TeeLogger:
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        self.encoding = "utf-8"

    def write(self, message: str) -> None:
        self.terminal.write(message)
        if "\r" not in message:
            self.log.write(message)
            self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def fileno(self) -> int:
        return self.terminal.fileno()


class HierarchicalDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        num_levels: int,
        total_classes: int,
        level_offsets: list[int],
        max_length: int,
        truncate_mode: str,
        strict: bool = True,
    ):
        self.data: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.num_levels = num_levels
        self.total_classes = total_classes
        self.level_offsets = level_offsets
        self.level_classes = [
            (level_offsets[i + 1] if i + 1 < len(level_offsets) else total_classes) - level_offsets[i]
            for i in range(num_levels)
        ]
        self.max_length = max_length
        self.truncate_mode = truncate_mode
        self.strict = strict
        self.level_tokens = LEVEL_TOKENS[:num_levels]
        self.level_token_ids = tokenizer.convert_tokens_to_ids(self.level_tokens)

        if any(pid is None or pid == tokenizer.unk_token_id for pid in self.level_token_ids):
            raise ValueError("Some level special tokens are not recognized by the tokenizer.")

        skipped = 0
        total_lines = 0
        with open(data_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                total_lines += 1
                try:
                    item = json.loads(line)
                    if "query" not in item:
                        raise ValueError("missing query")
                    if not isinstance(item["query"], str):
                        raise ValueError(f"query must be str, got {type(item['query']).__name__}")
                    if "label" not in item:
                        raise ValueError("missing label")
                    labels = item["label"]
                    if not isinstance(labels, list) or len(labels) != num_levels:
                        got_len = len(labels) if isinstance(labels, list) else "n/a"
                        raise ValueError(f"label must be a list with {num_levels} levels, got length {got_len}")

                    label_mask = item.get("label_mask")
                    if label_mask is None:
                        label_mask = [1] * num_levels
                    if not isinstance(label_mask, list) or len(label_mask) != num_levels:
                        got_len = len(label_mask) if isinstance(label_mask, list) else "n/a"
                        raise ValueError(f"label_mask must be a list with {num_levels} levels, got length {got_len}")
                    normalized_labels = []
                    for level_idx, label_indices in enumerate(labels):
                        indices = label_indices if isinstance(label_indices, list) else [label_indices]
                        normalized_level = []
                        for label_id in indices:
                            if isinstance(label_id, bool) or not isinstance(label_id, int):
                                raise ValueError(f"level {level_idx + 1} label id must be int, got {label_id!r}")
                            if not (0 <= label_id < self.level_classes[level_idx]):
                                raise ValueError(
                                    f"level {level_idx + 1} label id {label_id} out of range "
                                    f"[0, {self.level_classes[level_idx]})"
                                )
                            if label_id in normalized_level:
                                raise ValueError(f"level {level_idx + 1} duplicate label id {label_id}")
                            normalized_level.append(label_id)
                        normalized_labels.append(normalized_level)

                    normalized_mask = []
                    for mask in label_mask:
                        if isinstance(mask, bool):
                            normalized_mask.append(int(mask))
                        elif mask in (0, 1):
                            normalized_mask.append(int(mask))
                        else:
                            raise ValueError("label_mask values must be 0 or 1")

                    for level_idx, (level_labels, mask) in enumerate(zip(normalized_labels, normalized_mask)):
                        if mask == 0 and len(level_labels) > 0:
                            raise ValueError(
                                f"level {level_idx + 1} has label_mask=0 but non-empty labels"
                            )

                    self.data.append({"query": item["query"], "labels": normalized_labels, "label_mask": normalized_mask})
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    message = f"Malformed sample at {data_path}:{line_no}: {exc}"
                    if strict:
                        raise ValueError(message) from exc
                    skipped += 1
                    if skipped <= 5:
                        print(f"Skipping {message}")
        if len(self.data) == 0:
            raise ValueError(f"No valid samples loaded from {data_path}")
        if skipped:
            print(
                f"Loaded {len(self.data)} samples from {data_path} ({total_lines} non-empty lines, skipped {skipped} malformed).")
        else:
            print(f"Loaded {len(self.data)} samples from {data_path} ({total_lines} non-empty lines).")

    def __len__(self) -> int:
        return len(self.data)

    def _truncate(self, token_ids: list[int], max_query_len: int) -> list[int]:
        if len(token_ids) <= max_query_len:
            return token_ids
        if self.truncate_mode == "head":
            return token_ids[:max_query_len]

        head_len = max_query_len // 2
        tail_len = max_query_len - head_len
        return token_ids[:head_len] + token_ids[-tail_len:]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        query_tokens = self.tokenizer.encode(item["query"], add_special_tokens=False)

        max_query_len = self.max_length - len(self.level_token_ids)
        if max_query_len <= 0:
            raise ValueError("max_length is too small for level tokens.")
        query_tokens = self._truncate(query_tokens, max_query_len)

        input_ids = query_tokens + self.level_token_ids
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len

        labels = torch.zeros(self.num_levels, self.total_classes, dtype=torch.float32)
        label_mask = torch.tensor(item["label_mask"], dtype=torch.float32)
        for level_idx, label_indices in enumerate(item["labels"]):
            indices = label_indices if isinstance(label_indices, list) else [label_indices]
            level_offset = self.level_offsets[level_idx]
            for label_id in indices:
                if not (0 <= label_id < self.level_classes[level_idx]):
                    raise ValueError(
                        f"Dataset item {idx} has level {level_idx + 1} label id {label_id} out of range "
                        f"[0, {self.level_classes[level_idx]})"
                    )
                global_id = level_offset + label_id
                labels[level_idx, global_id] = 1.0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": labels,
            "label_mask": label_mask,
            "dataset_idx": torch.tensor(idx, dtype=torch.long),
        }


def collate_fn(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([feature[key] for feature in features]) for key in features[0]}


class LevelHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        intermediate_size = max(hidden_size // 4, 256)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HierarchicalTeacher(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: AutoTokenizer,
        num_levels: int,
        level_classes: list[int],
        level_offsets: list[int],
        hidden_layer_idx: int,
        use_hidden_states: bool,
        dropout: float,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_levels = num_levels
        self.level_classes = level_classes
        self.level_offsets = level_offsets
        self.hidden_layer_idx = hidden_layer_idx
        self.use_hidden_states = use_hidden_states
        self.level_token_ids = tokenizer.convert_tokens_to_ids(LEVEL_TOKENS[:num_levels])

        hidden_size = backbone.config.hidden_size
        self.level_heads = nn.ModuleList([LevelHead(hidden_size, n, dropout) for n in level_classes])
        self.global_head = LevelHead(hidden_size * num_levels, sum(level_classes), dropout)
        self.post_ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        target_dtype = getattr(backbone, "dtype", torch.float32)
        self.level_heads.to(target_dtype)
        self.global_head.to(target_dtype)
        self.post_ln.to(target_dtype)

    def _extract_level_features(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> list[torch.Tensor]:
        features = []
        batch_size = input_ids.size(0)
        for token_id in self.level_token_ids:
            per_sample = []
            for b in range(batch_size):
                pos = (input_ids[b] == token_id).nonzero(as_tuple=True)[0]
                if len(pos) == 0:
                    raise RuntimeError(f"Missing level token id {token_id} in sample {b}")
                per_sample.append(hidden_states[b, pos[0], :])
            features.append(torch.stack(per_sample, dim=0))
        return features

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.backbone.gradient_checkpointing_enable()
            else:
                self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels=None, **_: Any) -> dict[str, Any]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.use_hidden_states,
            output_attentions=False,
            return_dict=True,
        )
        if self.use_hidden_states:
            hidden_state_idx = (
                self.hidden_layer_idx
                if self.hidden_layer_idx < 0
                else self.hidden_layer_idx + 1
            )
            hidden = outputs.hidden_states[hidden_state_idx]
        else:
            hidden = outputs.last_hidden_state
        level_features = self._extract_level_features(hidden, input_ids)

        logits_list = []
        global_features = []
        for i, feat in enumerate(level_features):
            feat = self.dropout(self.post_ln(feat))
            global_features.append(feat)
            logits_list.append(self.level_heads[i](feat))
        global_logits = self.global_head(torch.cat(global_features, dim=-1))

        return {
            "logits_list": logits_list,
            "global_logits": global_logits,
            "logits_l1": logits_list[0] if self.num_levels >= 1 else None,
            "logits_l2": logits_list[1] if self.num_levels >= 2 else None,
            "logits_global": global_logits,
            "level_features": level_features,
            "labels": labels,
        }


@dataclass
class EvalCache:
    best_fixed_macro_f1: float = -1.0
    best_fixed_micro_f1: float = -1.0
    best_tuned_macro_f1: float = -1.0
    best_global_macro_f1: float = -1.0
    best_global_micro_f1: float = -1.0
    best_early_stop_metric: float | None = None
    early_stop_bad_count: int = 0


class HierarchicalTeacherTrainer(Trainer):
    def __init__(
        self,
        num_levels: int,
        level_classes: list[int],
        level_offsets: list[int],
        level_loss_weights: list[float],
        local_loss_weight: float,
        global_loss_weight: float,
        fixed_threshold: float,
        threshold_min: float,
        threshold_max: float,
        threshold_step: float,
        *args: Any,
        tune_threshold: bool = True,
        early_stopping_patience: int = 0,
        **kwargs: Any,
    ):
        self.save_tokenizer = kwargs.pop("tokenizer", None)
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.num_levels = num_levels
        self.level_classes = level_classes
        self.level_offsets = level_offsets
        self.level_loss_weights = level_loss_weights
        self.local_loss_weight = local_loss_weight
        self.global_loss_weight = global_loss_weight
        self.evaluate_local = local_loss_weight > 0
        self.evaluate_global = global_loss_weight > 0
        self.fixed_threshold = fixed_threshold
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_step = threshold_step
        self.tune_threshold = tune_threshold
        self.early_stopping_patience = early_stopping_patience
        self.eval_cache = EvalCache()
        self.loss_tracker = {
            "task_loss": 0.0,
            "local_task_loss": 0.0,
            **{f"level{i + 1}_loss": 0.0 for i in range(num_levels)},
        }
        if self.evaluate_global:
            self.loss_tracker["global_loss"] = 0.0
            for i in range(num_levels):
                self.loss_tracker[f"global_level{i + 1}_loss"] = 0.0
        self.loss_steps = 0

        if not self.evaluate_local and not self.evaluate_global:
            raise ValueError("local/global loss weights cannot both be zero")

        if not self.evaluate_local and hasattr(self.model, "level_heads"):
            for param in self.model.level_heads.parameters():
                param.requires_grad_(False)

        if not self.evaluate_global and hasattr(self.model, "global_head"):
            for param in self.model.global_head.parameters():
                param.requires_grad_(False)

    def _save_named_best(self, dirname: str, metric_name: str, metric_value: float, metrics: dict[str, float]) -> None:
        if not self.args.should_save or not self.is_world_process_zero():
            return

        save_dir = os.path.join(self.args.output_dir, dirname)
        self.save_model(save_dir)

        payload = {
            "selected_by": metric_name,
            "selected_value": metric_value,
            "global_step": int(self.state.global_step),
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        with open(os.path.join(save_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved {dirname} at step {self.state.global_step}: {metric_name}={metric_value:.6f}")

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        if output_dir is None:
            output_dir = self.args.output_dir
        if not self.is_world_process_zero():
            return

        os.makedirs(output_dir, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)

        if not hasattr(model, "backbone") or not hasattr(model.backbone, "save_pretrained"):
            raise TypeError("Expected HierarchicalTeacher.backbone to be a PEFT model with save_pretrained().")

        save_kwargs: dict[str, Any] = {
            "safe_serialization": getattr(self.args, "save_safetensors", True),
        }
        if "save_embedding_layers" in inspect.signature(model.backbone.save_pretrained).parameters:
            save_kwargs["save_embedding_layers"] = False
        model.backbone.save_pretrained(output_dir, **save_kwargs)

        classifier_state = {
            "level_heads": model.level_heads.state_dict(),
            "global_head": model.global_head.state_dict(),
            "post_ln": model.post_ln.state_dict(),
            "num_levels": self.num_levels,
            "level_classes": self.level_classes,
            "level_offsets": self.level_offsets,
            "global_loss_weight": self.global_loss_weight,
            "has_trained_local_heads": self.evaluate_local,
            "has_trained_global_head": self.evaluate_global,
        }
        torch.save(classifier_state, os.path.join(output_dir, "classifier_head.pt"))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        if self.save_tokenizer is not None:
            self.save_tokenizer.save_pretrained(output_dir)

    def _maybe_stop_early(self, metrics: dict[str, float]) -> None:
        if self.early_stopping_patience <= 0:
            return

        metric_name = self.args.metric_for_best_model
        if metric_name is None:
            metric_name = "eval_loss"
        if metric_name not in metrics and not metric_name.startswith("eval_"):
            metric_name = f"eval_{metric_name}"
        if metric_name not in metrics:
            print(f"Early stopping skipped: metric {metric_name} was not found.")
            return

        current = float(metrics[metric_name])
        greater_is_better = self.args.greater_is_better
        if greater_is_better is None:
            greater_is_better = not metric_name.endswith("loss")

        best = self.eval_cache.best_early_stop_metric
        improved = best is None or (current > best if greater_is_better else current < best)

        if improved:
            self.eval_cache.best_early_stop_metric = current
            self.eval_cache.early_stop_bad_count = 0
            print(f"Early stopping metric improved: {metric_name}={current:.6f}")
            return

        self.eval_cache.early_stop_bad_count += 1
        print(
            f"Early stopping patience: {self.eval_cache.early_stop_bad_count}/"
            f"{self.early_stopping_patience} without improving {metric_name} "
            f"(best={best:.6f}, current={current:.6f})"
        )
        if self.eval_cache.early_stop_bad_count >= self.early_stopping_patience:
            self.control.should_training_stop = True
            print(f"Early stopping triggered at step {self.state.global_step}.")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        label_mask = inputs.pop("label_mask", None)
        inputs.pop("dataset_idx", None)
        outputs = model(**inputs)

        local_logits_parts = []
        local_label_parts = []
        local_mask_parts = []
        local_level_losses = []
        global_label_parts = []
        global_mask_parts = []
        global_level_slices = []
        for i in range(self.num_levels):
            start = self.level_offsets[i]
            end = start + self.level_classes[i]
            level_labels = labels[:, i, start:end].float()
            level_logits = outputs["logits_list"][i].float()
            local_logits_parts.append(level_logits)
            local_label_parts.append(level_labels)
            if self.evaluate_global:
                global_label_parts.append(level_labels)
                global_level_slices.append((i, start, end))
            if label_mask is not None:
                curr_mask = label_mask[:, i].float()
                local_mask_parts.append(curr_mask.unsqueeze(1).expand(-1, self.level_classes[i]))
                if self.evaluate_global:
                    global_mask_parts.append(curr_mask.unsqueeze(1).expand(-1, self.level_classes[i]))

        local_logits = torch.cat(local_logits_parts, dim=-1)
        local_labels = torch.cat(local_label_parts, dim=-1)
        local_loss_matrix = F.binary_cross_entropy_with_logits(
            local_logits,
            local_labels,
            reduction="none",
        )
        if label_mask is not None:
            local_mask = torch.cat(local_mask_parts, dim=-1).float()
            local_task_loss = (local_loss_matrix * local_mask).sum() / local_mask.sum().clamp_min(1.0)
        else:
            local_mask = None
            local_task_loss = local_loss_matrix.mean()

        offset = 0
        for i in range(self.num_levels):
            width = self.level_classes[i]
            level_loss_matrix = local_loss_matrix[:, offset:offset + width]
            if local_mask is not None:
                level_mask = local_mask[:, offset:offset + width]
                level_loss = (level_loss_matrix * level_mask).sum() / level_mask.sum().clamp_min(1.0)
            else:
                level_loss = level_loss_matrix.mean()
            local_level_losses.append(level_loss)
            self.loss_tracker[f"level{i + 1}_loss"] += level_loss.detach().item()
            offset += width
        if self.evaluate_global:
            global_labels = torch.cat(global_label_parts, dim=-1)
            global_loss_matrix = F.binary_cross_entropy_with_logits(
                outputs["global_logits"].float(),
                global_labels,
                reduction="none",
            )
            if label_mask is not None:
                global_mask = torch.cat(global_mask_parts, dim=-1).float()
                global_loss = (global_loss_matrix * global_mask).sum() / (
                        global_mask.sum().clamp_min(1.0)
                )
            else:
                global_loss = global_loss_matrix.mean()
            for i, start, end in global_level_slices:
                global_level_matrix = global_loss_matrix[:, start:end]
                if label_mask is not None:
                    global_level_mask = label_mask[:, i].float().unsqueeze(1).expand(-1, self.level_classes[i])
                    global_level_loss = (global_level_matrix * global_level_mask).sum() / (
                            global_level_mask.sum().clamp_min(1.0)
                    )
                else:
                    global_level_loss = global_level_matrix.mean()
                self.loss_tracker[f"global_level{i + 1}_loss"] += global_level_loss.detach().item()
        else:
            global_loss = local_task_loss.new_zeros(())

        task_loss = self.local_loss_weight * local_task_loss + self.global_loss_weight * global_loss
        self.loss_tracker["task_loss"] += task_loss.detach().item()
        self.loss_tracker["local_task_loss"] += local_task_loss.detach().item()
        if self.evaluate_global:
            self.loss_tracker["global_loss"] += global_loss.detach().item()
        self.loss_steps += 1
        if self.evaluate_global:
            outputs["loss_global"] = global_loss.detach()
        for i, level_loss in enumerate(local_level_losses):
            outputs[f"loss_l{i + 1}"] = level_loss.detach()
        if local_level_losses:
            outputs["loss_l1"] = local_level_losses[0].detach()
        if len(local_level_losses) > 1:
            outputs["loss_l2"] = local_level_losses[1].detach()
        return (task_loss, outputs) if return_outputs else task_loss

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if self.loss_steps > 0:
            for key in self.loss_tracker:
                logs[key] = self.loss_tracker[key] / self.loss_steps
                self.loss_tracker[key] = 0.0
            self.loss_steps = 0
        super().log(logs, *args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        all_logits = {i: [] for i in range(self.num_levels)}
        all_global_logits = {i: [] for i in range(self.num_levels)}
        all_labels = {i: [] for i in range(self.num_levels)}
        all_flat_logits = []
        all_flat_global_logits = []
        all_flat_labels = []
        all_flat_mask = []
        level_loss_sums = [0.0 for _ in range(self.num_levels)]
        level_loss_counts = [0 for _ in range(self.num_levels)]
        global_level_loss_sums = [0.0 for _ in range(self.num_levels)]
        global_level_loss_counts = [0 for _ in range(self.num_levels)]
        local_loss_sum = 0.0
        local_loss_count = 0
        global_loss_sum = 0.0
        global_loss_count = 0

        # for batch in dataloader:
        progress_bar = tqdm(
            dataloader,
            desc=f"{metric_key_prefix} evaluating",
            total=len(dataloader),
            disable=not self.is_world_process_zero() or self.args.disable_tqdm,
            dynamic_ncols=True,
        )
        for batch in progress_bar:
            batch = self._prepare_inputs(batch)
            labels = batch["labels"]
            label_mask = batch.get("label_mask")
            model_inputs = {k: v for k, v in batch.items() if k not in {"labels", "label_mask", "dataset_idx"}}
            with torch.inference_mode():
                outputs = self.model(**model_inputs)

            global_label_parts = []
            global_mask_parts = []
            for i in range(self.num_levels):
                start = self.level_offsets[i]
                end = start + self.level_classes[i]
                global_label_parts.append(labels[:, i, start:end].detach().float())
                if label_mask is not None:
                    level_mask = label_mask[:, i].detach().float()
                else:
                    level_mask = torch.ones(labels.size(0), device=labels.device, dtype=torch.float32)
                global_mask_parts.append(level_mask.unsqueeze(1).expand(-1, self.level_classes[i]))
            flat_logits = torch.cat([outputs["logits_list"][i].detach().float() for i in range(self.num_levels)],
                                    dim=-1)
            global_labels = torch.cat(global_label_parts, dim=-1)
            global_mask = torch.cat(global_mask_parts, dim=-1)
            if self.evaluate_global:
                gathered_flat_logits, gathered_global_logits, gathered_global_labels, gathered_global_mask = self.accelerator.gather_for_metrics(
                    (
                        flat_logits.detach().float(),
                        outputs["global_logits"].detach().float(),
                        global_labels.detach().float(),
                        global_mask.detach().float(),
                    )
                )
            else:
                gathered_flat_logits, gathered_global_labels, gathered_global_mask = self.accelerator.gather_for_metrics(
                    (
                        flat_logits.detach().float(),
                        global_labels.detach().float(),
                        global_mask.detach().float(),
                    )
                )
            all_flat_logits.append(gathered_flat_logits.cpu().numpy())
            all_flat_labels.append(gathered_global_labels.cpu().numpy())
            all_flat_mask.append(gathered_global_mask.cpu().numpy())
            local_loss_matrix = F.binary_cross_entropy_with_logits(
                gathered_flat_logits.float(),
                gathered_global_labels.float(),
                reduction="none",
            )
            valid_local_count = gathered_global_mask.sum()
            if valid_local_count.item() > 0:
                local_loss_sum += (local_loss_matrix * gathered_global_mask).sum().detach().item()
                local_loss_count += int(valid_local_count.detach().item())
            if self.evaluate_global:
                all_flat_global_logits.append(gathered_global_logits.cpu().numpy())
                global_loss_matrix = F.binary_cross_entropy_with_logits(
                    gathered_global_logits.float(),
                    gathered_global_labels.float(),
                    reduction="none",
                )
                valid_global_count = gathered_global_mask.sum()
                if valid_global_count.item() > 0:
                    global_loss_sum += (global_loss_matrix * gathered_global_mask).sum().detach().item()
                    global_loss_count += int(valid_global_count.detach().item())

            gather_items = []
            for i in range(self.num_levels):
                start = self.level_offsets[i]
                end = start + self.level_classes[i]
                gather_items.append(outputs["logits_list"][i].detach().float())
                gather_items.append(labels[:, i, start:end].detach().float())
                if label_mask is not None:
                    gather_items.append(label_mask[:, i].detach().float())
                else:
                    gather_items.append(torch.ones(labels.size(0), device=labels.device, dtype=torch.float32))
                if self.evaluate_global:
                    gather_items.append(outputs["global_logits"][:, start:end].detach().float())

            gathered_items = self.accelerator.gather_for_metrics(tuple(gather_items))
            stride = 4 if self.evaluate_global else 3
            for i in range(self.num_levels):
                offset = stride * i
                gathered_logits = gathered_items[offset]
                gathered_labels = gathered_items[offset + 1]
                gathered_mask = gathered_items[offset + 2].bool()
                gathered_logits = gathered_logits[gathered_mask]
                gathered_labels = gathered_labels[gathered_mask]
                if gathered_logits.numel() > 0:
                    per_sample_loss = F.binary_cross_entropy_with_logits(
                        gathered_logits.float(),
                        gathered_labels.float(),
                        reduction="none",
                    ).mean(dim=1)
                    level_loss_sums[i] += per_sample_loss.sum().detach().item()
                    level_loss_counts[i] += int(per_sample_loss.numel())
                all_logits[i].append(gathered_logits.cpu().numpy())
                if self.evaluate_global:
                    gathered_global_logits = gathered_items[offset + 3]
                    gathered_global_logits = gathered_global_logits[gathered_mask]
                    if gathered_global_logits.numel() > 0:
                        global_per_sample_loss = F.binary_cross_entropy_with_logits(
                            gathered_global_logits.float(),
                            gathered_labels.float(),
                            reduction="none",
                        ).mean(dim=1)
                        global_level_loss_sums[i] += global_per_sample_loss.sum().detach().item()
                        global_level_loss_counts[i] += int(global_per_sample_loss.numel())
                    all_global_logits[i].append(gathered_global_logits.cpu().numpy())
                all_labels[i].append(gathered_labels.cpu().numpy())

        metrics: dict[str, float] = {}
        for i in range(self.num_levels):
            if level_loss_counts[i] == 0:
                continue
            level_loss = level_loss_sums[i] / level_loss_counts[i]
            metrics[f"{metric_key_prefix}_level{i + 1}_loss"] = level_loss
        eval_task_loss = local_loss_sum / local_loss_count if local_loss_count > 0 else 0.0
        metrics[f"{metric_key_prefix}_local_task_loss"] = eval_task_loss
        if self.evaluate_global:
            for i in range(self.num_levels):
                if global_level_loss_counts[i] == 0:
                    continue
                metrics[f"{metric_key_prefix}_global_level{i + 1}_loss"] = (
                        global_level_loss_sums[i] / global_level_loss_counts[i]
                )
            eval_global_loss = global_loss_sum / global_loss_count if global_loss_count > 0 else 0.0
            metrics[f"{metric_key_prefix}_global_loss"] = eval_global_loss
            eval_total_loss = self.local_loss_weight * eval_task_loss + self.global_loss_weight * eval_global_loss
        else:
            eval_total_loss = self.local_loss_weight * eval_task_loss
        metrics[f"{metric_key_prefix}_task_loss"] = eval_total_loss
        metrics[f"{metric_key_prefix}_loss"] = eval_total_loss

        fixed_macro_values = []
        fixed_micro_values = []
        tuned_macro_values = []
        tuned_micro_values = []

        thresholds = np.arange(self.threshold_min, self.threshold_max + 1e-9, self.threshold_step)
        for i in range(self.num_levels):
            if not all_logits[i] or sum(x.shape[0] for x in all_logits[i]) == 0:
                print(f"Level {i + 1}: skipped because no valid labels are available.")
                continue
            logits = np.concatenate(all_logits[i], axis=0)
            trues = np.concatenate(all_labels[i], axis=0)

            fixed_preds = (logits > self.fixed_threshold).astype(np.int32)
            fixed_macro = f1_score(trues, fixed_preds, average="macro", zero_division=0)
            fixed_micro = f1_score(trues, fixed_preds, average="micro", zero_division=0)


            level_id = i + 1
            metrics[f"{metric_key_prefix}_level{level_id}_fixed_macro_f1"] = fixed_macro
            metrics[f"{metric_key_prefix}_level{level_id}_fixed_micro_f1"] = fixed_micro


            fixed_macro_values.append(fixed_macro)
            fixed_micro_values.append(fixed_micro)

            if self.tune_threshold:
                best_threshold = self.fixed_threshold
                best_macro = -1.0
                best_micro_at_macro = 0.0
                for threshold in thresholds:
                    preds = (logits > threshold).astype(np.int32)
                    macro = f1_score(trues, preds, average="macro", zero_division=0)
                    if macro > best_macro:
                        best_macro = macro
                        best_threshold = float(threshold)
                        best_micro_at_macro = f1_score(trues, preds, average="micro", zero_division=0)

                metrics[f"{metric_key_prefix}_level{level_id}_tuned_macro_f1"] = best_macro
                metrics[f"{metric_key_prefix}_level{level_id}_tuned_micro_f1"] = best_micro_at_macro
                metrics[f"{metric_key_prefix}_level{level_id}_best_threshold"] = best_threshold
                tuned_macro_values.append(best_macro)
                tuned_micro_values.append(best_micro_at_macro)
            #     print(
            #         f"Level {level_id}: fixed macro/micro={fixed_macro:.4f}/{fixed_micro:.4f}, "
            #         f"tuned macro/micro={best_macro:.4f}/{best_micro_at_macro:.4f}, "
            #         f"best_threshold={best_threshold:.2f}"
            #     )
            # else:
            #     print(f"Level {level_id}: fixed macro/micro={fixed_macro:.4f}/{fixed_micro:.4f}")

        if all_flat_logits:
            flat_logits = np.concatenate(all_flat_logits, axis=0)
            flat_trues = np.concatenate(all_flat_labels, axis=0).astype(np.int32)
            flat_mask = np.concatenate(all_flat_mask, axis=0).astype(bool)

            flat_fixed_preds = (flat_logits > self.fixed_threshold).astype(np.int32)
            flat_fixed_macro = masked_multilabel_f1(flat_trues, flat_fixed_preds, flat_mask, average="macro")
            flat_fixed_micro = masked_multilabel_f1(flat_trues, flat_fixed_preds, flat_mask, average="micro")



            metrics[f"{metric_key_prefix}_flat_macro_f1"] = flat_fixed_macro
            metrics[f"{metric_key_prefix}_flat_micro_f1"] = flat_fixed_micro

            if self.tune_threshold:
                flat_best_threshold = self.fixed_threshold
                flat_best_macro = -1.0
                flat_best_micro_at_macro = 0.0
                for threshold in thresholds:
                    flat_preds = (flat_logits > threshold).astype(np.int32)
                    macro = masked_multilabel_f1(flat_trues, flat_preds, flat_mask, average="macro")
                    if macro > flat_best_macro:
                        flat_best_macro = macro
                        flat_best_threshold = float(threshold)
                        flat_best_micro_at_macro = masked_multilabel_f1(
                            flat_trues,
                            flat_preds,
                            flat_mask,
                            average="micro",
                        )

                metrics[f"{metric_key_prefix}_flat_tuned_macro_f1"] = flat_best_macro
                metrics[f"{metric_key_prefix}_flat_tuned_micro_f1"] = flat_best_micro_at_macro
                metrics[f"{metric_key_prefix}_flat_best_threshold"] = flat_best_threshold
            #     print(
            #         f"Flat Local: fixed macro/micro={flat_fixed_macro:.4f}/{flat_fixed_micro:.4f}, "
            #         f"tuned macro/micro={flat_best_macro:.4f}/{flat_best_micro_at_macro:.4f}, "
            #         f"best_threshold={flat_best_threshold:.2f}"
            #     )
            # else:
            #     print(f"Flat Local: fixed macro/micro={flat_fixed_macro:.4f}/{flat_fixed_micro:.4f}")

        global_fixed_macro_values = []
        global_fixed_micro_values = []
        global_tuned_macro_values = []
        global_tuned_micro_values = []
        for i in range(self.num_levels):
            if not all_global_logits[i] or sum(x.shape[0] for x in all_global_logits[i]) == 0:
                continue
            logits = np.concatenate(all_global_logits[i], axis=0)
            trues = np.concatenate(all_labels[i], axis=0)

            fixed_preds = (logits > self.fixed_threshold).astype(np.int32)
            fixed_macro = f1_score(trues, fixed_preds, average="macro", zero_division=0)
            fixed_micro = f1_score(trues, fixed_preds, average="micro", zero_division=0)


            level_id = i + 1
            metrics[f"{metric_key_prefix}_global_level{level_id}_fixed_macro_f1"] = fixed_macro
            metrics[f"{metric_key_prefix}_global_level{level_id}_fixed_micro_f1"] = fixed_micro

            global_fixed_macro_values.append(fixed_macro)
            global_fixed_micro_values.append(fixed_micro)

            if self.tune_threshold:
                best_threshold = self.fixed_threshold
                best_macro = -1.0
                best_micro_at_macro = 0.0
                for threshold in thresholds:
                    preds = (logits > threshold).astype(np.int32)
                    macro = f1_score(trues, preds, average="macro", zero_division=0)
                    if macro > best_macro:
                        best_macro = macro
                        best_threshold = float(threshold)
                        best_micro_at_macro = f1_score(trues, preds, average="micro", zero_division=0)

                metrics[f"{metric_key_prefix}_global_level{level_id}_tuned_macro_f1"] = best_macro
                metrics[f"{metric_key_prefix}_global_level{level_id}_tuned_micro_f1"] = best_micro_at_macro
                metrics[f"{metric_key_prefix}_global_level{level_id}_best_threshold"] = best_threshold
                global_tuned_macro_values.append(best_macro)
                global_tuned_micro_values.append(best_micro_at_macro)
            #     print(
            #         f"Global Level {level_id}: fixed macro/micro={fixed_macro:.4f}/{fixed_micro:.4f}, "
            #         f"tuned macro/micro={best_macro:.4f}/{best_micro_at_macro:.4f}, "
            #         f"best_threshold={best_threshold:.2f}"
            #     )
            # else:
            #     print(f"Global Level {level_id}: fixed macro/micro={fixed_macro:.4f}/{fixed_micro:.4f}")

        if all_flat_global_logits:
            flat_global_logits = np.concatenate(all_flat_global_logits, axis=0)
            flat_trues = np.concatenate(all_flat_labels, axis=0).astype(np.int32)
            flat_mask = np.concatenate(all_flat_mask, axis=0).astype(bool)

            flat_global_fixed_preds = (flat_global_logits > self.fixed_threshold).astype(np.int32)
            flat_global_fixed_macro = masked_multilabel_f1(
                flat_trues,
                flat_global_fixed_preds,
                flat_mask,
                average="macro",
            )
            flat_global_fixed_micro = masked_multilabel_f1(
                flat_trues,
                flat_global_fixed_preds,
                flat_mask,
                average="micro",
            )



            metrics[f"{metric_key_prefix}_global_flat_macro_f1"] = flat_global_fixed_macro
            metrics[f"{metric_key_prefix}_global_flat_micro_f1"] = flat_global_fixed_micro

            if self.tune_threshold:
                flat_global_best_threshold = self.fixed_threshold
                flat_global_best_macro = -1.0
                flat_global_best_micro_at_macro = 0.0
                for threshold in thresholds:
                    flat_global_preds = (flat_global_logits > threshold).astype(np.int32)
                    macro = masked_multilabel_f1(flat_trues, flat_global_preds, flat_mask, average="macro")
                    if macro > flat_global_best_macro:
                        flat_global_best_macro = macro
                        flat_global_best_threshold = float(threshold)
                        flat_global_best_micro_at_macro = masked_multilabel_f1(
                            flat_trues,
                            flat_global_preds,
                            flat_mask,
                            average="micro",
                        )

                metrics[f"{metric_key_prefix}_global_flat_tuned_macro_f1"] = flat_global_best_macro
                metrics[f"{metric_key_prefix}_global_flat_tuned_micro_f1"] = flat_global_best_micro_at_macro
                metrics[f"{metric_key_prefix}_global_flat_best_threshold"] = flat_global_best_threshold
            #     print(
            #         f"Flat Global: fixed macro/micro={flat_global_fixed_macro:.4f}/{flat_global_fixed_micro:.4f}, "
            #         f"tuned macro/micro={flat_global_best_macro:.4f}/{flat_global_best_micro_at_macro:.4f}, "
            #         f"best_threshold={flat_global_best_threshold:.2f}"
            #     )
            # else:
            #     print(f"Flat Global: fixed macro/micro={flat_global_fixed_macro:.4f}/{flat_global_fixed_micro:.4f}")

        self.log(metrics)

        if metric_key_prefix == "eval":
            if self.evaluate_local:
                fixed_macro = metrics[f"{metric_key_prefix}_flat_macro_f1"]
                fixed_micro = metrics[f"{metric_key_prefix}_flat_micro_f1"]
                tuned_macro = metrics[f"{metric_key_prefix}_flat_tuned_macro_f1"]

                if fixed_macro > self.eval_cache.best_fixed_macro_f1:
                    self.eval_cache.best_fixed_macro_f1 = fixed_macro
                    self._save_named_best("best_fixed_macro", "eval_flat_macro_f1", fixed_macro, metrics)

                if fixed_micro > self.eval_cache.best_fixed_micro_f1:
                    self.eval_cache.best_fixed_micro_f1 = fixed_micro
                    self._save_named_best("best_fixed_micro", "eval_flat_micro_f1", fixed_micro, metrics)

                if tuned_macro > self.eval_cache.best_tuned_macro_f1:
                    self.eval_cache.best_tuned_macro_f1 = tuned_macro
                    self._save_named_best("best_tuned_macro", "eval_flat_tuned_macro_f1", tuned_macro, metrics)

            if self.evaluate_global:
                global_macro = metrics[f"{metric_key_prefix}_global_flat_macro_f1"]
                global_micro = metrics[f"{metric_key_prefix}_global_flat_micro_f1"]
                if global_macro > self.eval_cache.best_global_macro_f1:
                    self.eval_cache.best_global_macro_f1 = global_macro
                    self._save_named_best("best_global_macro", "eval_global_flat_macro_f1", global_macro, metrics)
                if global_micro > self.eval_cache.best_global_micro_f1:
                    self.eval_cache.best_global_micro_f1 = global_micro
                    self._save_named_best("best_global_micro", "eval_global_flat_micro_f1", global_micro, metrics)
            self.control = self.callback_handler.on_evaluate(
                self.args,
                self.state,
                self.control,
                metrics,
            )
            self._maybe_stop_early(metrics)
        return metrics


def main() -> None:
    args = parse_args()


    seed_everything(args.seed)

    dataset_cfg = DATASET_ZOO[args.dataset]
    model_path = args.model_path or MODEL_ZOO[args.model_key]
    data_dir = args.data_dir or dataset_cfg["path"]
    max_length = args.max_length or dataset_cfg["max_length"]

    train_path = os.path.join(data_dir, dataset_cfg["train_file"])
    dev_path = os.path.join(data_dir, dataset_cfg["dev_file"])

    model_name = Path(model_path).name
    output_dir = args.output_dir or f"/data1/lwq/agt_plus/output_plus/{args.dataset}_{model_name}_baseline_SFT"
    run_name = args.run_name or Path(output_dir).name
    os.makedirs(output_dir, exist_ok=True)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        sys.stdout = TeeLogger(os.path.join(output_dir, "training.log"))
        sys.stderr = sys.stdout

    meta = load_preprocess_meta(train_path)
    dev_meta = load_preprocess_meta(dev_path)
    assert_preprocess_meta_compatible(meta, dev_meta, "dev")
    num_levels = meta["num_levels"]
    if num_levels > len(LEVEL_TOKENS):
        raise ValueError(f"num_levels={num_levels} exceeds available LEVEL_TOKENS={len(LEVEL_TOKENS)}")
    level_classes = meta["level_classes"]
    total_classes = meta["total_classes"]
    level_offsets = build_level_offsets(level_classes)
    level_loss_weights = parse_level_loss_weights(args.level_loss_weights, num_levels)
    if args.early_stop_target == "local" and args.local_loss_weight <= 0:
        raise ValueError(
            "--early_stop_target local requires --local_loss_weight > 0"
        )
    if args.early_stop_target == "global":
        if args.global_loss_weight <= 0:
            raise ValueError("--early_stop_target global requires --global_loss_weight > 0.")
        early_stop_metric = "eval_global_flat_macro_f1"
    else:
        early_stop_metric = "eval_flat_macro_f1"

    print(f"Dataset={args.dataset}, train={train_path}, dev={dev_path}")
    print(f"Model={model_path}")
    print(f"Run name={run_name}")
    print(
        f"num_levels={num_levels}, level_classes={level_classes}, "
        f"level_loss_weights={level_loss_weights}, local_loss_weight={args.local_loss_weight}"
    )
    print(f"Early stopping target={args.early_stop_target}, metric={early_stop_metric}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base_tokenizer_vocab_size = len(tokenizer)
    num_added_level_tokens = add_level_special_tokens(
        tokenizer,
        LEVEL_TOKENS[:num_levels],
    )
    level_token_ids = tokenizer.convert_tokens_to_ids(LEVEL_TOKENS[:num_levels])
    level_tokens_are_new = (
            num_added_level_tokens == num_levels
            and all(isinstance(token_id, int) and token_id >= base_tokenizer_vocab_size for token_id in level_token_ids)
    )
    print(f"Actually added {num_added_level_tokens} new tokens")
    print(f"Level token ids: {level_token_ids}")
    print(
        f"Level tokens are new: {level_tokens_are_new} "
        f"(base_tokenizer_vocab_size={base_tokenizer_vocab_size}, tokenizer_vocab_size={len(tokenizer)})"
    )
    if not level_tokens_are_new:
        print(
            "Warning: not all level tokens look newly added. "
            "Check teacher_config.json before interpreting them as new anchor tokens."
        )

    train_dataset = HierarchicalDataset(
        train_path,
        tokenizer,
        num_levels,
        total_classes,
        level_offsets,
        max_length,
        args.truncate_mode,
        strict=True,
    )
    eval_dataset = HierarchicalDataset(
        dev_path,
        tokenizer,
        num_levels,
        total_classes,
        level_offsets,
        max_length,
        args.truncate_mode,
        strict=True,
    )

    can_use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    backbone = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if can_use_bf16 else torch.float16,
        trust_remote_code=True,
    )
    backbone.config.use_cache = False
    # backbone.resize_token_embeddings(len(tokenizer))
    current_num_embeddings = (
        backbone.get_input_embeddings().num_embeddings
    )

    target_num_embeddings = max(
        current_num_embeddings,
        len(tokenizer),
    )

    if target_num_embeddings != current_num_embeddings:
        backbone.resize_token_embeddings(
            target_num_embeddings
        )

    if any(token_id is None or token_id == tokenizer.unk_token_id for token_id in level_token_ids):
        raise ValueError("Some level special tokens are not recognized by the tokenizer.")
    if "trainable_token_indices" not in inspect.signature(LoraConfig.__init__).parameters:
        raise RuntimeError(
            "This PEFT version does not support LoraConfig(trainable_token_indices=...). "
            "Upgrade PEFT or use a Trainable Tokens configuration before running this experiment."
        )

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        trainable_token_indices={"embed_tokens": level_token_ids},
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()

    model = HierarchicalTeacher(
        backbone=backbone,
        tokenizer=tokenizer,
        num_levels=num_levels,
        level_classes=level_classes,
        level_offsets=level_offsets,
        hidden_layer_idx=args.student_layer_idx,
        use_hidden_states=args.use_hidden_states,
        dropout=args.dropout,
    )

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
        load_best_model_at_end=False,
        metric_for_best_model=early_stop_metric,
        greater_is_better=True,
        remove_unused_columns=False,
        report_to=[] if args.no_swanlab else ["swanlab"],
        run_name=run_name,
        dataloader_pin_memory=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )


    trainer = HierarchicalTeacherTrainer(
        num_levels=num_levels,
        level_classes=level_classes,
        level_offsets=level_offsets,
        level_loss_weights=level_loss_weights,
        local_loss_weight=args.local_loss_weight,
        global_loss_weight=args.global_loss_weight,
        fixed_threshold=args.fixed_threshold,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        early_stopping_patience=args.early_stopping_patience,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )

    if trainer.is_world_process_zero():
        print(
            "Trainer loss handling:",
            f"model_accepts_loss_kwargs={trainer.model_accepts_loss_kwargs},",
            f"gradient_accumulation_steps={trainer.args.gradient_accumulation_steps}",
        )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_model"))

    teacher_config = {
        "dataset": args.dataset,
        "model_path": model_path,
        "run_name": run_name,
        "num_levels": num_levels,
        "level_classes": level_classes,
        "level_offsets": level_offsets,
        "label_space_source": meta["meta"].get("label_space_source"),
        "layer_label2id": normalize_layer_label2id(meta["meta"]),
        "level_tokens": LEVEL_TOKENS[:num_levels],
        "level_token_ids": level_token_ids,
        "base_tokenizer_vocab_size": base_tokenizer_vocab_size,
        "tokenizer_vocab_size": len(tokenizer),
        "num_added_level_tokens": num_added_level_tokens,
        "level_tokens_are_new": level_tokens_are_new,
        "embedding_training": "trainable_token_indices",
        "hidden_layer_idx": args.student_layer_idx,
        "hidden_layer_indexing": "transformer_block_0_based",
        "hidden_states_index": args.student_layer_idx if args.student_layer_idx < 0 else args.student_layer_idx + 1,
        "use_hidden_states": args.use_hidden_states,
        "max_length": max_length,
        "truncate_mode": args.truncate_mode,
        "level_loss_weights": level_loss_weights,
        "local_loss_weight": args.local_loss_weight,
        "global_loss_weight": args.global_loss_weight,
        "early_stop_target": args.early_stop_target,
        "early_stop_metric": early_stop_metric,
        "fixed_threshold": args.fixed_threshold,
        "seed": args.seed,
        "data_seed": args.seed,
        "has_trained_local_heads": args.local_loss_weight > 0,
        "has_trained_global_head": args.global_loss_weight > 0,
    }
    if trainer.is_world_process_zero():
        for dirname in ["final_model", "best_fixed_macro", "best_fixed_micro", "best_tuned_macro", "best_global_macro", "best_global_micro"]:
            dir_path = os.path.join(output_dir, dirname)
            if os.path.isdir(dir_path):
                write_json(os.path.join(dir_path, "teacher_config.json"), teacher_config)

    final_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="final_eval")
    if trainer.is_world_process_zero():
        write_json(os.path.join(output_dir, "final_eval_metrics.json"), final_metrics)


if __name__ == "__main__":
    main()
