"""Generative HTC LoRA SFT using the existing baseline's data/meta/F1 conventions.

Runtime dependency: train_baseline_collator.py (left unchanged).
No extra hierarchy tokens, classifier, teacher cache, or TRL dependency.
See GENERATIVE_SFT.md for commands, decoding rules, and comparison limits.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shlex
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FORMAT_VERSION = "generative_htc_json_names_v1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truncate_text(ids: list[int], budget: int, mode: str) -> list[int]:
    # Same effective text budget and head/head_tail rule as HierarchicalDataset.
    if budget <= 0:
        raise ValueError("max_length must exceed the number of hierarchy levels")
    if len(ids) <= budget:
        return ids
    if mode == "head":
        return ids[:budget]
    if mode != "head_tail":
        raise ValueError(f"Unknown truncate_mode: {mode}")
    head = budget // 2
    return ids[:head] + ids[-(budget - head):]


class LabelCodec:
    """Fixed, level-specific label vocabulary. Never fills in missing ancestors."""

    def __init__(self, layer_label2id: list[dict[str, int]]):
        self.label2id = [{str(k): int(v) for k, v in level.items()} for level in layer_label2id]
        self.id2label = []
        self.offsets = []
        self.total_classes = 0
        for level in self.label2id:
            if not level or sorted(level.values()) != list(range(len(level))):
                raise ValueError("Each level must have unique contiguous label IDs starting at zero")
            self.offsets.append(self.total_classes)
            self.total_classes += len(level)
            self.id2label.append({v: k for k, v in level.items()})
        self.num_levels = len(self.label2id)
        if not self.num_levels:
            raise ValueError("Empty label space")

    def serialize(self, labels: list[list[int]]) -> tuple[str, list[tuple[int, int]]]:
        # Full JSON is tokenized at once. Character spans let us mask unknown
        # levels without tokenizing every level separately (BPE boundary changes).
        answer = "["
        spans = []
        for level, ids in enumerate(labels):
            if level:
                answer += ","
            start = len(answer)
            answer += json.dumps([self.id2label[level][i] for i in sorted(ids)], ensure_ascii=False)
            spans.append((start, len(answer)))
        return answer + "]", spans

    def encode_gold(self, labels: list[list[int]], mask: list[int]) -> tuple[list[int], list[int]]:
        gold, flat_mask = [0] * self.total_classes, []
        for level, ids in enumerate(labels):
            flat_mask.extend([mask[level]] * len(self.label2id[level]))
            for index in ids:
                gold[self.offsets[level] + index] = 1
        return gold, flat_mask

    def parse(self, text: str) -> tuple[list[int], list[int]]:
        """Return multihot and [malformed_json, unknown_labels, duplicate_labels]."""
        pred = [0] * self.total_classes
        try:
            obj = json.loads(text.strip())
        except (ValueError, TypeError):
            return pred, [1, 0, 0]
        if (not isinstance(obj, list) or len(obj) != self.num_levels
                or any(not isinstance(row, list) for row in obj)):
            return pred, [1, 0, 0]
        unknown, duplicate = 0, 0
        for level, row in enumerate(obj):
            seen = set()
            for name in row:
                if not isinstance(name, str) or name not in self.label2id[level]:
                    unknown += 1
                    continue
                if name in seen:
                    duplicate += 1
                seen.add(name)
                pred[self.offsets[level] + self.label2id[level][name]] = 1
        return pred, [0, unknown, duplicate]


def load_records(path: Path, codec: LabelCodec) -> list[dict]:
    """Match the original strict JSONL schema: query, label, optional label_mask."""
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row["query"], str):
                    raise ValueError("query must be a string")
                labels = row["label"]
                mask = row.get("label_mask", [1] * codec.num_levels)
                if not isinstance(labels, list) or len(labels) != codec.num_levels:
                    raise ValueError("label must contain one entry per level")
                if not isinstance(mask, list) or len(mask) != codec.num_levels or any(m not in (0, 1) for m in mask):
                    raise ValueError("label_mask must contain one 0/1 entry per level")
                normalized = []
                for level, values in enumerate(labels):
                    ids = values if isinstance(values, list) else [values]
                    if any(isinstance(i, bool) or not isinstance(i, int) or i not in codec.id2label[level] for i in ids):
                        raise ValueError(f"invalid label ID at level {level + 1}")
                    if len(set(ids)) != len(ids):
                        raise ValueError(f"duplicate label ID at level {level + 1}")
                    if not mask[level] and ids:
                        raise ValueError("label_mask=0 with non-empty labels")
                    normalized.append(sorted(ids))
                records.append({"query": row["query"], "labels": normalized,
                                "label_mask": [int(m) for m in mask], "line_number": line_number})
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Malformed sample at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"No samples in {path}")
    return records


def make_prompt(num_levels: int) -> tuple[str, str]:
    # These are ordinary strings encoded with the unchanged base tokenizer.
    prefix = (
        "Classify the following text into hierarchical labels. "
        f"Return only a JSON array containing {num_levels} arrays, "
        "one per level from root to leaf. Each array must contain all applicable "
        "label names at that level. Use [] if there are none. Do not explain.\n\nText:\n"
    )
    return prefix, "\n\nLabels:\n"


def encode_answer(tokenizer, codec: LabelCodec, record: dict) -> tuple[list[int], list[int]]:
    text, spans = codec.serialize(record["labels"])
    if all(record["label_mask"]):
        ids = tokenizer.encode(text, add_special_tokens=False)
        targets = list(ids)
    else:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer with offsets is required to mask label_mask=0 levels")
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        unknown = [span for span, active in zip(spans, record["label_mask"]) if not active]
        targets = [
            -100 if any(start < right and end > left for left, right in unknown) else token
            for token, (start, end) in zip(ids, encoded["offset_mapping"])
        ]
    # EOS is a real supervised token even when PAD and EOS have the same ID.
    return ids + [tokenizer.eos_token_id], targets + [tokenizer.eos_token_id]


class GenerativeDataset(Dataset):
    def __init__(self, path: Path, tokenizer, codec: LabelCodec, max_length: int,
                 truncate_mode: str, max_new_tokens: int, context_limit: int,
                 check_answer_budget: bool = True):
        self.path, self.tokenizer, self.codec = Path(path), tokenizer, codec
        self.records = load_records(self.path, codec)
        self.text_budget = max_length - codec.num_levels
        self.truncate_mode = truncate_mode
        if self.text_budget <= 0:
            raise ValueError("max_length must exceed num_levels")
        prefix, suffix = make_prompt(codec.num_levels)
        self.prefix = tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix = tokenizer.encode(suffix, add_special_tokens=False)
        required = len(self.prefix) + self.text_budget + len(self.suffix) + max_new_tokens
        if required > context_limit:
            raise ValueError(
                f"Prompt + matched text budget + generation budget needs {required} tokens, "
                f"model context limit is {context_limit}. Reduce --max_length/--max_new_tokens explicitly."
            )
        self.answers = []
        longest = 0
        # Only small completions are cached in RAM, not full vocab logits or full prompts.
        for record in self.records:
            answer = encode_answer(tokenizer, codec, record)
            longest = max(longest, len(answer[0]))
            if check_answer_budget and len(answer[0]) > max_new_tokens:
                raise ValueError(
                    f"{path}:{record['line_number']}: gold answer needs {len(answer[0])} tokens including EOS; "
                    f"--max_new_tokens={max_new_tokens}. Increase it using train/dev only; answers are never truncated."
                )
            self.answers.append(answer)
        self.stats = {"samples": len(self.records), "max_answer_tokens": longest, "text_budget": self.text_budget,
                      "masked_level_samples": sum(not all(x["label_mask"]) for x in self.records)}
        print(f"{self.path.name}: {self.stats}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        text = self.tokenizer.encode(record["query"], add_special_tokens=False)
        text = truncate_text(text, self.text_budget, self.truncate_mode)
        prompt = self.prefix + text + self.suffix
        answer, targets = self.answers[index]
        gold, mask = self.codec.encode_gold(record["labels"], record["label_mask"])
        return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + targets,
                "prompt_ids": prompt, "gold": gold, "class_mask": mask, "sample_idx": index}


class GenerativeCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        batch_size = len(items)
        width = max(len(item["input_ids"]) for item in items)
        prompt_width = max(len(item["prompt_ids"]) for item in items)
        batch = {
            "input_ids": torch.full((batch_size, width), self.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((batch_size, width), dtype=torch.long),
            "labels": torch.full((batch_size, width), -100, dtype=torch.long),
            "prompt_ids": torch.full((batch_size, prompt_width), self.pad_token_id, dtype=torch.long),
            "prompt_mask": torch.zeros((batch_size, prompt_width), dtype=torch.long),
        }
        for i, item in enumerate(items):
            length, prompt_length = len(item["input_ids"]), len(item["prompt_ids"])
            # Right padding for training; left padding for decoder-only generation.
            batch["input_ids"][i, :length] = torch.tensor(item["input_ids"])
            batch["attention_mask"][i, :length] = 1
            batch["labels"][i, :length] = torch.tensor(item["labels"])
            batch["prompt_ids"][i, -prompt_length:] = torch.tensor(item["prompt_ids"])
            batch["prompt_mask"][i, -prompt_length:] = 1
        for name in ("gold", "class_mask", "sample_idx"):
            batch[name] = torch.tensor([item[name] for item in items], dtype=torch.long)
        return batch


def decode_completions(tokenizer, generated: torch.Tensor) -> tuple[list[str], list[int]]:
    rows, missing_eos = [], []
    for row in generated.detach().cpu().tolist():
        try:
            end = row.index(tokenizer.eos_token_id)
            missing_eos.append(0)
        except ValueError:
            end = len(row)
            missing_eos.append(1)
        rows.append(row[:end])
    # Don't silently remove invalid special tokens from a generated label name.
    return tokenizer.batch_decode(rows, skip_special_tokens=False, clean_up_tokenization_spaces=False), missing_eos


def complete_json_prefix(text: str, num_levels: int) -> str | None:
    """Find a complete level-array answer starting at the first non-space character.

    A token may contain characters beyond the closing bracket, so generation and
    scoring use the same JSON boundary. Leading prose and incomplete arrays do
    not qualify. Label names are checked later by LabelCodec.parse.
    """
    candidate = text.lstrip()
    if not candidate.startswith("["):
        return None
    try:
        value, end = json.JSONDecoder().raw_decode(candidate)
    except ValueError:
        return None
    if (not isinstance(value, list) or len(value) != num_levels
            or any(not isinstance(level, list) for level in value)):
        return None
    return candidate[:end]


def make_json_stopping_criteria(tokenizer, num_levels: int, prompt_width: int):
    from transformers import StoppingCriteria

    class StopAfterCompleteJson(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            continuation = input_ids[:, prompt_width:].detach().cpu().tolist()
            texts = tokenizer.batch_decode(continuation, skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False)
            return torch.tensor([complete_json_prefix(text, num_levels) is not None for text in texts],
                                dtype=torch.bool, device=input_ids.device)

    return StopAfterCompleteJson()


def make_compute_metrics(codec: LabelCodec, masked_f1):
    def compute_metrics(result):
        preds, diagnostics, _, _ = result.predictions
        gold, mask = result.label_ids[:, 0], result.label_ids[:, 1]
        metrics = {
            "global_flat_micro_f1": masked_f1(gold, preds, mask, average="micro"),
            "global_flat_macro_f1": masked_f1(gold, preds, mask, average="macro"),
            "invalid_json_rate": float(np.mean(diagnostics[:, 0])),
            "unknown_label_rate": float(np.mean(diagnostics[:, 1] > 0)),
            "unknown_labels_per_sample": float(np.mean(diagnostics[:, 1])),
            "duplicate_label_rate": float(np.mean(diagnostics[:, 2] > 0)),
            "missing_eos_rate": float(np.mean(diagnostics[:, 3])),
            "complete_json_rate": float(np.mean(diagnostics[:, 4])),
        }
        for level, start in enumerate(codec.offsets):
            end = start + len(codec.label2id[level])
            for average in ("micro", "macro"):
                metrics[f"level{level+1}_{average}_f1"] = masked_f1(
                    gold[:, start:end], preds[:, start:end], mask[:, start:end], average=average)
        return metrics
    return compute_metrics


def build_trainer_class():
    from transformers import GenerationConfig, StoppingCriteriaList, Trainer

    class GenerativeTrainer(Trainer):
        def __init__(self, *args, label_codec, generation_tokenizer, max_new_tokens, **kwargs):
            super().__init__(*args, **kwargs)
            self.label_codec = label_codec
            self.generation_tokenizer = generation_tokenizer
            self.max_new_tokens = max_new_tokens
            # Standard mean CE per microbatch; Trainer applies grad accumulation.
            self.model_accepts_loss_kwargs = False
            base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
            self.selective_logits = "logits_to_keep" in inspect.signature(base.forward).parameters
            self.generation_options = GenerationConfig(
                max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                eos_token_id=generation_tokenizer.eos_token_id,
                pad_token_id=generation_tokenizer.pad_token_id,
                bos_token_id=generation_tokenizer.bos_token_id, use_cache=True,
            )

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if self.selective_logits:
                # Generate vocabulary logits only where some answer token needs
                # supervision. This is the SAME shifted CE, avoiding prompt-sized
                # [batch, sequence, 151k-vocabulary] tensors on Qwen3.
                targets = inputs["labels"][:, 1:]
                positions = (targets != -100).any(dim=0).nonzero(as_tuple=True)[0]
                outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                logits_to_keep=positions)
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
                    targets[:, positions].reshape(-1), ignore_index=-100,
                )
            else:
                outputs = model(**{name: inputs[name] for name in ("input_ids", "attention_mask", "labels")})
                loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            if prediction_loss_only:
                raise ValueError("Generative HTC evaluation requires generation, not prediction_loss_only")
            inputs = self._prepare_inputs(inputs)
            generator = self.accelerator.unwrap_model(model)
            with torch.no_grad(), self.compute_loss_context_manager():
                # Gold answer, label_mask, and gold labels NEVER enter generate().
                sequences = generator.generate(
                    input_ids=inputs["prompt_ids"], attention_mask=inputs["prompt_mask"],
                    generation_config=self.generation_options,
                    stopping_criteria=StoppingCriteriaList([make_json_stopping_criteria(
                        self.generation_tokenizer, self.label_codec.num_levels, inputs["prompt_ids"].shape[1])]),
                )
            continuation = sequences[:, inputs["prompt_ids"].shape[1]:]
            texts, no_eos = decode_completions(self.generation_tokenizer, continuation)
            predictions, diagnostics = [], []
            for text, truncated in zip(texts, no_eos):
                answer = complete_json_prefix(text, self.label_codec.num_levels)
                pred, diag = self.label_codec.parse(answer if answer is not None else text)
                predictions.append(pred)
                diagnostics.append(diag + [truncated, int(answer is not None)])
            device = inputs["gold"].device
            # Fixed lengths keep Trainer's cross-batch/process gather unambiguous.
            padded = torch.full((len(texts), self.max_new_tokens), self.generation_tokenizer.eos_token_id,
                                device=device, dtype=torch.long)
            padded[:, :continuation.shape[1]] = continuation
            result = (torch.tensor(predictions, device=device), torch.tensor(diagnostics, device=device),
                      padded, inputs["sample_idx"].unsqueeze(1))
            labels = torch.stack((inputs["gold"], inputs["class_mask"]), dim=1)
            return None, result, labels

    return GenerativeTrainer


def make_best_callback(output_dir: Path, run_config: dict):
    from transformers import TrainerCallback

    class SaveNamedBest(TrainerCallback):
        trainer = None

        def __init__(self):
            self.best = {"macro": -1.0, "micro": -1.0}
            # Named model metrics persist the additional micro checkpoint on resume.
            for average in self.best:
                path = output_dir / f"best_global_{average}" / "selection.json"
                if path.is_file():
                    self.best[average] = json.loads(path.read_text(encoding="utf-8"))["score"]

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            for average in self.best:
                key = f"eval_global_flat_{average}_f1"
                if key not in metrics or metrics[key] <= self.best[average]:
                    continue
                self.best[average] = float(metrics[key])
                path = output_dir / f"best_global_{average}"
                self.trainer.save_model(str(path))
                if self.trainer.is_world_process_zero():
                    write_json(path / "sft_config.json", run_config)
                    write_json(path / "selection.json", {"metric": key, "score": self.best[average],
                               "step": state.global_step, "epoch": state.epoch, "dev_metrics": metrics})
            return control

    return SaveNamedBest()


def parse_args(reference, argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=sorted(reference.DATASET_ZOO), default="bgc")
    p.add_argument("--model_key", choices=sorted(reference.MODEL_ZOO), default="qwen0.6b")
    p.add_argument("--model_path")
    p.add_argument("--data_dir")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name")
    p.add_argument("--gpu", help="Optional; otherwise respect CUDA_VISIBLE_DEVICES")
    p.add_argument("--max_length", type=int, help="Original baseline length; text budget=max_length-num_levels")
    p.add_argument("--truncate_mode", choices=["head", "head_tail"], default="head")
    p.add_argument("--max_new_tokens", type=int, default=512)
    # Same general training defaults as train_baseline_collator.py.
    for name, default in {"seed": 42, "train_batch": 8, "eval_batch": 8, "grad_acc": 4,
                          "logging_steps": 10, "eval_steps": 200, "save_steps": 200,
                          "save_total_limit": 3, "early_stopping_patience": 5,
                          "lora_r": 16, "lora_alpha": 32}.items():
        p.add_argument(f"--{name}", type=int, default=default)
    for name, default in {"epochs": 15., "lr": 8e-5, "weight_decay": .01, "warmup_ratio": .03,
                          "lora_dropout": .05}.items():
        p.add_argument(f"--{name}", type=float, default=default)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--do_test", action="store_true", help="Evaluate test ONCE with dev-macro-selected weights")
    p.add_argument("--resume_from_checkpoint", help="A checkpoint-* directory inside this output_dir")
    args = p.parse_args(argv)
    for name in ("train_batch", "eval_batch", "grad_acc", "logging_steps", "eval_steps", "save_steps",
                 "save_total_limit", "early_stopping_patience", "max_new_tokens", "lora_r", "lora_alpha"):
        if getattr(args, name) <= 0:
            p.error(f"--{name} must be positive")
    if args.epochs <= 0 or args.lr <= 0 or not 0 <= args.warmup_ratio <= 1:
        p.error("epochs/lr must be positive; warmup_ratio must be in [0,1]")
    if args.save_steps != args.eval_steps:
        p.error("Use save_steps=eval_steps so each dev selection has a resumable checkpoint")
    return args


def save_predictions(path: Path, prediction, dataset: GenerativeDataset):
    _, diagnostics, generated, indices = prediction.predictions
    texts, _ = decode_completions(dataset.tokenizer, torch.as_tensor(generated))
    order = sorted(range(len(indices)), key=lambda i: int(indices[i, 0]))
    with path.open("w", encoding="utf-8") as stream:
        for i in order:
            index = int(indices[i, 0])
            record = dataset.records[index]
            stream.write(json.dumps({"sample_idx": index, "source_line": record["line_number"],
                "generated": texts[i],
                "accepted_json": complete_json_prefix(texts[i], dataset.codec.num_levels),
                "gold": json.loads(dataset.codec.serialize(record["labels"])[0]),
                "label_mask": record["label_mask"], "diagnostics": diagnostics[i].tolist()},
                ensure_ascii=False) + "\n")


def main():
    # Reuse the user's verified original paths, metadata checks and F1 implementation.
    import train_baseline_collator as reference

    args = parse_args(reference)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from peft import LoraConfig, TaskType, get_peft_model
    import peft
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainingArguments

    reference.seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    if args.resume_from_checkpoint:
        checkpoint = Path(args.resume_from_checkpoint).resolve()
        if checkpoint.parent != output_dir.resolve() or not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("resume_from_checkpoint must be a checkpoint-* with trainer_state.json in output_dir")
    elif int(os.environ.get("RANK", "0")) == 0 and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty; use a new directory or --resume_from_checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0:
        sys.stdout = reference.TeeLogger(str(output_dir / "training.log"))
        sys.stderr = sys.stdout
    cfg = reference.DATASET_ZOO[args.dataset]
    model_path = args.model_path or reference.MODEL_ZOO[args.model_key]
    data_dir = Path(args.data_dir or cfg["path"])
    paths = {split: data_dir / cfg[f"{split}_file"] for split in ("train", "dev", "test")}
    meta = reference.load_preprocess_meta(str(paths["train"]))
    reference.assert_preprocess_meta_compatible(meta, reference.load_preprocess_meta(str(paths["dev"])), "dev")
    codec = LabelCodec(reference.normalize_layer_label2id(meta["meta"]))
    if codec.num_levels != meta["num_levels"]:
        raise ValueError("Metadata num_levels does not match layer_label2id")
    max_length = args.max_length if args.max_length is not None else cfg["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Base tokenizer must have an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if any(token in tokenizer.get_vocab() for token in reference.LEVEL_TOKENS):
        raise ValueError("Tokenizer contains custom level tokens; point --model_path to the original base checkpoint")
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    context_limit = getattr(model_config, "max_position_embeddings", None)
    if not isinstance(context_limit, int) or context_limit <= 0:
        raise ValueError("Cannot determine model context limit from max_position_embeddings")
    dataset_kwargs = dict(tokenizer=tokenizer, codec=codec, max_length=max_length,
                          truncate_mode=args.truncate_mode, max_new_tokens=args.max_new_tokens,
                          context_limit=context_limit)
    train_dataset = GenerativeDataset(paths["train"], **dataset_kwargs)
    dev_dataset = GenerativeDataset(paths["dev"], **dataset_kwargs)
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    protocol = {
        "format": FORMAT_VERSION, "dataset": args.dataset, "model_path": str(model_path),
        "layer_label2id": codec.label2id, "max_length": max_length,
        "text_budget": train_dataset.text_budget, "truncate_mode": args.truncate_mode,
        "prompt": make_prompt(codec.num_levels), "max_new_tokens": args.max_new_tokens,
        "new_special_tokens": 0, "decoding": "greedy_stop_at_complete_json_v2",
        "invalid_json": "all_zero_predictions", "unknown_labels": "drop_and_report",
        "hierarchy_postprocessing": "none", "loss": "answer_only_token_ce_microbatch_mean",
        "masked_levels": "mask_overlapping_answer_tokens; retain structural_separators_and_eos",
        "lora_targets": LORA_TARGETS, "dtype": str(dtype),
        "source_hashes": {split: file_sha256(paths[split]) for split in ("train", "dev")},
        "training": {key: getattr(args, key) for key in (
            "seed", "train_batch", "eval_batch", "grad_acc", "epochs", "lr", "weight_decay", "warmup_ratio",
            "eval_steps", "save_steps", "early_stopping_patience", "lora_r", "lora_alpha", "lora_dropout",
            "gradient_checkpointing")},
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }
    # JSON round-trip makes tuple/list comparison stable after resuming.
    protocol = json.loads(json.dumps(protocol))
    config_path = output_dir / "sft_config.json"
    if args.resume_from_checkpoint:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous["protocol"] != protocol:
            raise ValueError("Resume protocol differs from sft_config.json; use a new run directory")
    run_config = {"protocol": protocol, "args": vars(args), "train_stats": train_dataset.stats,
                  "dev_stats": dev_dataset.stats, "versions": {"torch": torch.__version__,
                  "transformers": transformers.__version__, "peft": peft.__version__},
                  "reference_sha256": file_sha256(Path(reference.__file__)),
                  "script_sha256": file_sha256(Path(__file__))}
    if int(os.environ.get("RANK", "0")) == 0:
        write_json(config_path, run_config)
        (output_dir / ("resume_command.txt" if args.resume_from_checkpoint else "command.txt")).write_text(
            shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
    # No resize_token_embeddings, trainable level-token indices, or classifier head.
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **{dtype_key: dtype})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=LORA_TARGETS, bias="none"))
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    ta_parameters = inspect.signature(TrainingArguments).parameters
    ta = dict(output_dir=str(output_dir), seed=args.seed, data_seed=args.seed,
        per_device_train_batch_size=args.train_batch, per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.grad_acc, num_train_epochs=args.epochs,
        learning_rate=args.lr, weight_decay=args.weight_decay, bf16=bf16, fp16=cuda and not bf16,
        logging_steps=args.logging_steps, eval_steps=args.eval_steps, save_strategy="steps",
        save_steps=args.save_steps, save_total_limit=args.save_total_limit, load_best_model_at_end=True,
        metric_for_best_model="eval_global_flat_macro_f1", greater_is_better=True,
        remove_unused_columns=False, label_names=["labels"], eval_accumulation_steps=1,
        report_to=[] if args.no_swanlab else ["swanlab"], run_name=args.run_name or output_dir.name,
        dataloader_pin_memory=cuda, gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False)
    ta["eval_strategy" if "eval_strategy" in ta_parameters else "evaluation_strategy"] = "steps"
    if "warmup_ratio" in ta_parameters:
        ta["warmup_ratio"] = args.warmup_ratio
    else:
        # New Transformers versions accept fractional warmup_steps in [0,1).
        # Use integer steps for unambiguous compatibility, including ratio=1.
        per_rank_batches = math.ceil(math.ceil(len(train_dataset) / protocol["world_size"]) / args.train_batch)
        total_steps = math.ceil(math.ceil(per_rank_batches / args.grad_acc) * args.epochs)
        ta["warmup_steps"] = math.ceil(total_steps * args.warmup_ratio)
    if "restore_callback_states_from_checkpoint" in ta_parameters:
        ta["restore_callback_states_from_checkpoint"] = True
    training_args = TrainingArguments(**ta)
    trainer_cls = build_trainer_class()
    named_best = make_best_callback(output_dir, run_config)
    tokenizer_key = "processing_class" if "processing_class" in inspect.signature(transformers.Trainer.__init__).parameters else "tokenizer"
    trainer = trainer_cls(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset,
        data_collator=GenerativeCollator(tokenizer.pad_token_id), label_codec=codec, generation_tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens, compute_metrics=make_compute_metrics(codec, reference.masked_multilabel_f1),
        callbacks=[named_best, EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        **{tokenizer_key: tokenizer})
    named_best.trainer = trainer
    print(f"Generative SFT: {model_path}; text_budget={train_dataset.text_budget}; "
          f"eval every {args.eval_steps} optimizer steps; greedy JSON-complete stopping, "
          f"max_new_tokens={args.max_new_tokens}")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    if trainer.state.best_model_checkpoint is None:
        # Very short smoke runs may finish before the first scheduled evaluation.
        trainer.evaluate()
    # Trainer has restored the checkpoint selected by dev macro F1 (if any).
    prediction = trainer.predict(dev_dataset, metric_key_prefix="best_dev")
    trainer.save_metrics("best_dev", prediction.metrics)
    if trainer.is_world_process_zero():
        save_predictions(output_dir / "best_dev_predictions.jsonl", prediction, dev_dataset)
    if args.do_test:
        # Test labels never influence training, hyperparameters, early stopping or selection.
        reference.assert_preprocess_meta_compatible(meta, reference.load_preprocess_meta(str(paths["test"])), "test")
        # Never tune the generation budget using test-label lengths. Keep all
        # reference labels even if their serialization exceeds the fixed budget.
        test_dataset = GenerativeDataset(paths["test"], check_answer_budget=False, **dataset_kwargs)
        prediction = trainer.predict(test_dataset, metric_key_prefix="test")
        trainer.save_metrics("test", prediction.metrics)
        if trainer.is_world_process_zero():
            save_predictions(output_dir / "test_predictions.jsonl", prediction, test_dataset)
    print(f"Done. Dev-macro-selected adapter: {output_dir / 'best_global_macro'}")


if __name__ == "__main__":
    main()
