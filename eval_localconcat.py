"""
 CUDA_VISIBLE_DEVICES=1  python eval_global.py \
  --checkpoint_dir /data1/lwq/agt_plus/output/wos_Qwen3-8B-Base_baseline_SFT/best_global_macro \
  --split test \
  --eval_batch 16

 CUDA_VISIBLE_DEVICES=1  python eval_global.py \
  --checkpoint_dir /data1/lwq/agt_plus/output/wos_qwen0.6b_local_only/best_fixed_micro \
  --split test \
  --eval_batch 16

 CUDA_VISIBLE_DEVICES=1  python eval_global.py \
  --checkpoint_dir /data1/lwq/agt_plus/output/wos_Qwen3-0.6B-Base_baseline_SFT/best_fixed_macro \
  --split test \
  --eval_batch 16
 CUDA_VISIBLE_DEVICES=0  python eval_localconcat.py \
  --checkpoint_dir /data1/lwq/agt_plus/output_plus/_qwen0.6b_local0_global1-ir2e-4/best_global_macro \
  --split test \
  --eval_batch 16
 CUDA_VISIBLE_DEVICES=2  python eval_localconcat.py \
  --checkpoint_dir /data1/lwq/agt_plus/output_plus/rcv1_qwen0.6b_local0_global1-lr2e-4/best_global_macro \
  --split test \
  --eval_batch 16
 CUDA_VISIBLE_DEVICES=1  python eval_localconcat.py \
  --checkpoint_dir /data1/lwq/agt_plus/output_plus/wos_qwen0.6b_local0.5_global1-lr2e-4/best_global_macro \
  --split test \
  --eval_batch 16
CUDA_VISIBLE_DEVICES=1  python eval_localconcat.py \
  --checkpoint_dir /data1/lwq/agt_plus/output_plus/rcv1_qwen8b_local0.5_global1_collator_new/best_global_macro \
  --split test \
  --eval_batch 16
CUDA_VISIBLE_DEVICES=2  python eval_localconcat.py \
  --checkpoint_dir /data1/lwq/agt_plus/output_distill/global-only/wos_Qwen3-0.6B-global-only_g1-l0-1logit-0.1feature-10att-epoch10/best_global_macro \
  --split test \
  --eval_batch 16
CUDA_VISIBLE_DEVICES=1  python eval_localconcat.py \
  --checkpoint_dir /data1/lwq/agt_plus/output_distill/global-only/wos_Qwen3-0.6B-global-only_g1-l0-1logit-0.1feature-0.2att-epoch10/best_global_macro \
  --split test \
  --eval_batch 16
CUDA_VISIBLE_DEVICES=1  python eval_localconcat.py \
  --checkpoint_dir /data1/lwq/agt_plus/output_distill/global-only/wos_Qwen3-0.6B-global-only_g1-l0-1logit-0.1feature-2att-epoch10/best_global_macro \
  --split test \
  --eval_batch 16
CUDA_VISIBLE_DEVICES=1  python eval_localconcat.py \
  --checkpoint_dir /data1/lwq/agt_plus/output_distill/global-only/wos_Qwen3-0.6B-global-only_g1-l0-1logit-0.1feature-10att-epoch10/best_global_macro \
  --split test \
  --eval_batch 16
CUDA_VISIBLE_DEVICES=1  python eval_localconcat.py \
  --checkpoint_dir /data1/lwq/agt_plus/output_distill/global+local/wos_Qwen3-0.6B-global+local_feature0.05-only-epoch10/best_global_macro \
  --split test \
  --eval_batch 16

CUDA_VISIBLE_DEVICES=1 python eval_global.py \
  --checkpoint_dir /data1/lwq/agt_plus/output_plus/rcv1_qwen8b_local0.1_global1_collator/best_global_macro \
  --dataset rcv1 \
  --split dev \
  --eval_batch 8 \
  --output_json /data1/lwq/agt_plus/output_plus/rcv1_qwen8b_local0.1_global1_collator/best_global_macro/new_dev_metrics.json

"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, TrainingArguments

from train_baseline import (
    DATASET_ZOO,
    LEVEL_TOKENS,
    MODEL_ZOO,
    HierarchicalDataset,
    HierarchicalTeacher,
    HierarchicalTeacherTrainer,
    add_level_special_tokens,
    assert_preprocess_meta_compatible,
    build_level_offsets,
    collate_fn,
    load_preprocess_meta,
    normalize_layer_label2id,
    parse_level_loss_weights,
    seed_everything,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a hierarchical teacher checkpoint on a held-out split.")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing adapter_model and classifier_head.pt.")
    parser.add_argument("--dataset", default=None, choices=sorted(DATASET_ZOO))
    parser.add_argument("--model_key", default=None, choices=sorted(MODEL_ZOO))
    parser.add_argument("--model_path", default=None, help="Override base model path.")
    parser.add_argument("--data_dir", default=None, help="Override dataset path.")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--output_json", default=None)

    # parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_batch", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--truncate_mode", choices=["head", "head_tail"], default=None)
    parser.add_argument("--student_layer_idx", type=int, default=None)
    parser.add_argument("--use_hidden_states", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--global_loss_weight", type=float, default=None)
    parser.add_argument("--local_loss_weight", type=float, default=None)
    parser.add_argument("--level_loss_weights", default=None)
    parser.add_argument("--threshold_min", type=float, default=-2.0)
    parser.add_argument("--threshold_max", type=float, default=1.0)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--fixed_threshold", type=float, default=None)
    return parser.parse_args()


def read_json_if_exists(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_teacher_config(checkpoint_dir: str) -> dict[str, Any]:
    config = read_json_if_exists(os.path.join(checkpoint_dir, "teacher_config.json"))
    if config:
        return config

    parent_config = read_json_if_exists(os.path.join(os.path.dirname(checkpoint_dir), "teacher_config.json"))
    return parent_config


def get_split_path(dataset_cfg: dict[str, Any], data_dir: str, split: str) -> str:
    file_key = {"train": "train_file", "dev": "dev_file", "test": "test_file"}[split]
    return os.path.join(data_dir, dataset_cfg[file_key])


def keep_final_metric(name: str) -> bool:
    if "_tuned_" in name or name.endswith("_best_threshold"):
        return False
    if not name.endswith(("_macro_f1", "_micro_f1")):
        return False
    return True

def main() -> None:
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed_everything(args.seed)

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    teacher_config = find_teacher_config(checkpoint_dir)

    dataset_name = args.dataset or teacher_config.get("dataset")
    if dataset_name is None:
        raise ValueError("Pass --dataset or use a checkpoint with teacher_config.json.")
    dataset_cfg = DATASET_ZOO[dataset_name]

    model_path = args.model_path or teacher_config.get("model_path")
    if model_path is None:
        if args.model_key is None:
            raise ValueError("Pass --model_path/--model_key or use a checkpoint with teacher_config.json.")
        model_path = MODEL_ZOO[args.model_key]

    data_dir = args.data_dir or dataset_cfg["path"]
    train_path = get_split_path(dataset_cfg, data_dir, "train")
    eval_path = get_split_path(dataset_cfg, data_dir, args.split)

    meta = load_preprocess_meta(train_path)
    eval_meta = load_preprocess_meta(eval_path)
    assert_preprocess_meta_compatible(meta, eval_meta, args.split)
    expected_label_space_source = teacher_config.get("label_space_source")
    current_label_space_source = meta["meta"].get("label_space_source")
    if expected_label_space_source is not None and expected_label_space_source != current_label_space_source:
        raise RuntimeError(
            "Checkpoint label_space_source mismatch: "
            f"checkpoint={expected_label_space_source}, eval_data={current_label_space_source}"
        )
    expected_layer_label2id = teacher_config.get("layer_label2id")
    if expected_layer_label2id is not None:
        if normalize_layer_label2id({"layer_label2id": expected_layer_label2id}) != normalize_layer_label2id(
                meta["meta"]):
            raise RuntimeError("Checkpoint layer_label2id does not match current eval data label space.")
    else:
        print("Warning: teacher_config has no layer_label2id; checkpoint label-id semantics cannot be fully verified.")
    num_levels = int(teacher_config.get("num_levels", meta["num_levels"]))
    if num_levels > len(LEVEL_TOKENS):
        raise ValueError(f"num_levels={num_levels} exceeds available LEVEL_TOKENS={len(LEVEL_TOKENS)}")
    level_classes = list(teacher_config.get("level_classes", meta["level_classes"]))
    total_classes = int(sum(level_classes))
    level_offsets = list(teacher_config.get("level_offsets", build_level_offsets(level_classes)))

    max_length = args.max_length or int(teacher_config.get("max_length", dataset_cfg["max_length"]))
    truncate_mode = args.truncate_mode or teacher_config.get("truncate_mode", "head")
    hidden_layer_idx = args.student_layer_idx
    if hidden_layer_idx is None:
        hidden_layer_idx = int(teacher_config.get("hidden_layer_idx", -1))
    use_hidden_states = args.use_hidden_states or bool(teacher_config.get("use_hidden_states", False))
    fixed_threshold = args.fixed_threshold
    if fixed_threshold is None:
        fixed_threshold = float(teacher_config.get("fixed_threshold", 0.0))
    global_loss_weight_known = args.global_loss_weight is not None or "global_loss_weight" in teacher_config
    global_loss_weight = args.global_loss_weight
    if global_loss_weight is None:
        global_loss_weight = float(teacher_config.get("global_loss_weight", 0.0))
    local_loss_weight = args.local_loss_weight
    if local_loss_weight is None:
        local_loss_weight = float(teacher_config.get("local_loss_weight", 1.0))

    raw_level_weights = args.level_loss_weights
    if raw_level_weights is None and "level_loss_weights" in teacher_config:
        level_loss_weights = [float(x) for x in teacher_config["level_loss_weights"]]
    else:
        level_loss_weights = parse_level_loss_weights(raw_level_weights, num_levels)


    tokenizer_from_checkpoint = os.path.exists(os.path.join(checkpoint_dir, "tokenizer_config.json"))
    tokenizer_path = checkpoint_dir if tokenizer_from_checkpoint else model_path
    tokenizer_source = "checkpoint" if tokenizer_from_checkpoint else "base model"

    print(f"Loading tokenizer from {tokenizer_source}: {tokenizer_path}")

    if not tokenizer_from_checkpoint:
        print(
            "Warning: checkpoint has no saved tokenizer; falling back to base model tokenizer. "
            "Level token ids will be checked if teacher_config.json contains level_token_ids."
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if not tokenizer_from_checkpoint:
        add_level_special_tokens(
            tokenizer,
            LEVEL_TOKENS[:num_levels],
        )

    actual_level_token_ids = tokenizer.convert_tokens_to_ids(LEVEL_TOKENS[:num_levels])
    print(f"Eval level token ids: {actual_level_token_ids}")
    expected_level_token_ids = teacher_config.get("level_token_ids")
    if expected_level_token_ids is not None and list(expected_level_token_ids) != actual_level_token_ids:
        raise RuntimeError(
            "Level token ids mismatch between checkpoint config and eval tokenizer: "
            f"checkpoint={expected_level_token_ids}, eval={actual_level_token_ids}"
        )
    if expected_level_token_ids is None:
        print("Warning: teacher_config has no level_token_ids; eval tokenizer ids cannot be verified.")

    eval_dataset = HierarchicalDataset(
        eval_path,
        tokenizer,
        num_levels,
        total_classes,
        level_offsets,
        max_length,
        truncate_mode,
        strict=True,
    )

    can_use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    base_model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if can_use_bf16 else torch.float16,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    # base_model.resize_token_embeddings(len(tokenizer))
    current_num_embeddings = (
        base_model.get_input_embeddings().num_embeddings
    )

    target_num_embeddings = max(
        current_num_embeddings,
        len(tokenizer),
    )

    if target_num_embeddings != current_num_embeddings:
        base_model.resize_token_embeddings(
            target_num_embeddings
        )
    backbone = PeftModel.from_pretrained(base_model, checkpoint_dir)

    model = HierarchicalTeacher(
        backbone=backbone,
        tokenizer=tokenizer,
        num_levels=num_levels,
        level_classes=level_classes,
        level_offsets=level_offsets,
        hidden_layer_idx=hidden_layer_idx,
        use_hidden_states=use_hidden_states,
        dropout=args.dropout,
    )

    classifier_path = os.path.join(checkpoint_dir, "classifier_head.pt")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Missing classifier head: {classifier_path}")
    classifier_state = torch.load(classifier_path, map_location="cpu")
    required_classifier_meta = ["num_levels", "level_classes", "level_offsets"]
    missing_classifier_meta = [key for key in required_classifier_meta if key not in classifier_state]
    if missing_classifier_meta:
        raise RuntimeError(
            f"Checkpoint classifier_head.pt is missing required metadata: {missing_classifier_meta}"
        )
    if int(classifier_state["num_levels"]) != num_levels:
        raise RuntimeError(
            f"Checkpoint num_levels mismatch: checkpoint={classifier_state['num_levels']}, eval={num_levels}"
        )
    if list(classifier_state["level_classes"]) != level_classes:
        raise RuntimeError(
            f"Checkpoint level_classes mismatch: checkpoint={classifier_state['level_classes']}, eval={level_classes}"
        )
    if list(classifier_state["level_offsets"]) != level_offsets:
        raise RuntimeError(
            f"Checkpoint level_offsets mismatch: checkpoint={classifier_state['level_offsets']}, eval={level_offsets}"
        )
    if args.global_loss_weight is None and "global_loss_weight" in classifier_state:
        global_loss_weight = float(classifier_state["global_loss_weight"])
        global_loss_weight_known = True
    if args.local_loss_weight is None and "local_loss_weight" in classifier_state:
        local_loss_weight = float(classifier_state["local_loss_weight"])
    if not global_loss_weight_known and "global_head" in classifier_state and "has_trained_global_head" not in classifier_state:
        raise RuntimeError(
            "Cannot determine whether checkpoint global_head was trained. "
            "Pass --global_loss_weight 0 for local-only evaluation or provide a checkpoint with teacher_config.json."
        )
    model.level_heads.load_state_dict(classifier_state["level_heads"])
    model.post_ln.load_state_dict(classifier_state["post_ln"])
    has_trained_global_head = bool(
        classifier_state.get(
            "has_trained_global_head",
            "global_head" in classifier_state and global_loss_weight > 0,
        )
    )
    if has_trained_global_head:
        if "global_head" not in classifier_state:
            raise RuntimeError("Checkpoint metadata says global_head was trained, but global_head state is missing.")
        model.global_head.load_state_dict(classifier_state["global_head"])
    elif global_loss_weight > 0:
        raise RuntimeError(
            "Checkpoint has no trained global_head, but global_loss_weight > 0. "
            "Global evaluation is invalid."
        )
    else:
        print("Skipping global metrics because this checkpoint is local-only.")

    output_dir = os.path.join(checkpoint_dir, f"{args.split}_eval_tmp")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=args.eval_batch,
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=True,
        bf16=can_use_bf16,
        fp16=not can_use_bf16 and torch.cuda.is_available(),
    )

    trainer = HierarchicalTeacherTrainer(
        num_levels=num_levels,
        level_classes=level_classes,
        level_offsets=level_offsets,
        level_loss_weights=level_loss_weights,
        local_loss_weight=local_loss_weight,
        global_loss_weight=global_loss_weight,
        fixed_threshold=fixed_threshold,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        tune_threshold=False,
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )

    metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix=args.split)
    report_metrics = {k: float(v) for k, v in metrics.items() if keep_final_metric(k)}
    payload = {
        "checkpoint_dir": checkpoint_dir,
        "dataset": dataset_name,
        "split": args.split,
        "model_path": model_path,
        "eval_path": eval_path,
        "tokenizer_source": tokenizer_source,
        "tokenizer_path": tokenizer_path,
        "level_token_ids": actual_level_token_ids,
        "fixed_threshold": float(fixed_threshold),
        "local_loss_weight": float(local_loss_weight),
        "global_loss_weight": float(global_loss_weight),
        "level_loss_weights": [float(x) for x in level_loss_weights],
        "has_trained_global_head": bool(has_trained_global_head),
        "metrics": report_metrics,
    }

    output_json = args.output_json or os.path.join(checkpoint_dir, f"{args.split}_metrics.json")
    write_json(output_json, payload)
    print("\nFixed-threshold F1 results")
    for name in sorted(report_metrics):
        print(f"{name}: {report_metrics[name]:.6f}")
    print(f"Wrote metrics to {output_json}")


if __name__ == "__main__":
    main()
