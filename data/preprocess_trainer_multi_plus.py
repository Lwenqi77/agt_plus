#!/usr/bin/env python3
"""
rcv1_dev_ctr_com_0603.json
python preprocess_trainer_multi_plus.py --dataset rcv1 --split train --base-dir /data1/lwq/agt_plus
python preprocess_trainer_multi.py --dataset wos --split dev --base-dir /data1/lwq/agt_plus
python preprocess_trainer_multi.py --dataset wos --split test --base-dir /data1/lwq/agt_plus
python preprocess_trainer_multi.pyy --all --base-dir /data1/lwq/agt_agt
生成脚本：
rcv1_train_trainer_multi.json
rcv1_dev_trainer_multi.json
rcv1_test_trainer_multi.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


DEFAULT_NUM_LEVELS = {
    "wos": 2,
    "rcv1": None,
    "nyt": None,
}

OUTPUT_FILES = {
    "train": "{dataset}_train_trainer_multi.json",
    "dev": "{dataset}_dev_trainer_multi.json",
    "test": "{dataset}_test_trainer_multi.json",
}


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    if not samples:
        raise RuntimeError(f"No samples found in {path}")
    return samples


def load_value_dict(data_dir: Path) -> dict[Any, Any]:
    path = data_dir / "value_dict.pt"
    if not path.exists():
        raise FileNotFoundError(f"value_dict.pt not found: {path}")
    return torch.load(path, weights_only=False)


def load_tree(data_dir: Path, dataset: str, tree_path: Path | None = None) -> Any:
    candidates = []
    if tree_path is not None:
        candidates.append(tree_path)
    candidates.extend(
        [
            data_dir / f"{dataset}_tree.json",
            data_dir / f"{dataset}_tree.pt",
            data_dir / f"{dataset}_tree.pth",
            data_dir / "tree.json",
            data_dir / "tree.pt",
            data_dir / "tree.pth",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".json":
                return load_json(path)
            return torch.load(path, weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load tree file {path}: {exc}") from exc
    raise FileNotFoundError(f"No tree file found under {data_dir}")


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return None


def infer_text(sample: dict[str, Any]) -> str:
    for key in ("token", "query", "text", "content", "document", "body", "sentence"):
        value = sample.get(key)
        if value:
            return str(value)
    raise KeyError(f"Cannot find text field. sample keys={list(sample.keys())}")


def normalize_label_field(label_field: Any) -> list[list[int]]:
    if label_field is None or label_field == []:
        return []

    if isinstance(label_field, list) and label_field and isinstance(label_field[0], list):
        raw_groups = label_field
    else:
        raw_groups = [label_field]

    groups: list[list[int]] = []
    for group in raw_groups:
        if not isinstance(group, list):
            group = [group]
        ids = [as_int(x) for x in group]
        ids = [x for x in ids if x is not None]
        deduped = []
        seen = set()
        for x in ids:
            if x not in seen:
                deduped.append(x)
                seen.add(x)
        if deduped:
            groups.append(deduped)
    return groups


def build_tree_depth(tree_obj: Any) -> dict[str, int]:
    name_to_depth: dict[str, int] = {}

    def visit(name: Any, obj: Any, depth: int) -> None:
        key = str(name)
        if key not in name_to_depth or depth < name_to_depth[key]:
            name_to_depth[key] = depth
        if isinstance(obj, dict):
            for child_name, child_obj in obj.items():
                visit(child_name, child_obj, depth + 1)
        elif isinstance(obj, list):
            for child in obj:
                if isinstance(child, (dict, list)):
                    visit(child, {}, depth + 1)
                else:
                    visit(child, {}, depth + 1)

    if not isinstance(tree_obj, dict):
        raise TypeError(f"Expected tree object to be dict, got {type(tree_obj)}")

    for root_name, root_obj in tree_obj.items():
        if isinstance(root_obj, dict):
            for child_name, child_obj in root_obj.items():
                visit(child_name, child_obj, 0)
        else:
            visit(root_name, root_obj, 0)
    return name_to_depth


def value_lookup(value_dict: dict[Any, Any], raw_id: int) -> str | None:
    candidates = [raw_id, str(raw_id)]
    for key in candidates:
        if key in value_dict:
            return str(value_dict[key])

    # Some value_dict files are label_name -> id. Fall back to reverse lookup.
    raw_id_str = str(raw_id)
    for key, value in value_dict.items():
        if value == raw_id or str(value) == raw_id_str:
            return str(key)
    return None


def labels_to_layers(
    raw_label_groups: list[list[int]],
    value_dict: dict[Any, Any],
    name_to_depth: dict[str, int],
    num_levels: int,
) -> list[list[str]]:
    layers: list[list[str]] = [[] for _ in range(num_levels)]
    for group in raw_label_groups:
        for raw_id in group:
            name = value_lookup(value_dict, raw_id)
            if name is None:
                continue
            depth = name_to_depth.get(name)
            if depth is None:
                depth = name_to_depth.get(str(raw_id))
            if depth is None or depth >= num_levels:
                continue
            if name not in layers[depth]:
                layers[depth].append(name)
    return layers


def build_label_space(layered_samples: list[list[list[str]]], num_levels: int) -> tuple[list[dict[str, int]], list[dict[int, str]]]:
    layer_label2id: list[dict[str, int]] = []
    layer_id2label: list[dict[int, str]] = []
    for level in range(num_levels):
        names = sorted({name for sample in layered_samples for name in sample[level]})
        l2i = {name: idx for idx, name in enumerate(names)}
        i2l = {idx: name for name, idx in l2i.items()}
        layer_label2id.append(l2i)
        layer_id2label.append(i2l)
    return layer_label2id, layer_id2label


def value_dict_label_names(value_dict: dict[Any, Any], name_to_depth: dict[str, int], num_levels: int) -> list[str]:
    names: list[str] = []
    seen = set()

    def add_candidate(value: Any) -> None:
        name = str(value)
        depth = name_to_depth.get(name)
        if depth is None or depth >= num_levels or name in seen:
            return
        names.append(name)
        seen.add(name)

    for key, value in value_dict.items():
        key_as_id = as_int(key)
        value_as_id = as_int(value)

        if key_as_id is not None and value_as_id is None:
            add_candidate(value)
        elif key_as_id is None and value_as_id is not None:
            add_candidate(key)
        else:
            add_candidate(value)
            add_candidate(key)

    return names


def build_label_space_from_names(
    names: list[str],
    name_to_depth: dict[str, int],
    num_levels: int,
) -> tuple[list[dict[str, int]], list[dict[int, str]]]:
    names_by_level = [[] for _ in range(num_levels)]
    seen_by_level = [set() for _ in range(num_levels)]
    for name in names:
        depth = name_to_depth.get(name)
        if depth is None or depth >= num_levels or name in seen_by_level[depth]:
            continue
        names_by_level[depth].append(name)
        seen_by_level[depth].add(name)

    layer_label2id: list[dict[str, int]] = []
    layer_id2label: list[dict[int, str]] = []
    for level_names in names_by_level:
        sorted_names = sorted(level_names)
        l2i = {name: idx for idx, name in enumerate(sorted_names)}
        i2l = {idx: name for name, idx in l2i.items()}
        layer_label2id.append(l2i)
        layer_id2label.append(i2l)
    return layer_label2id, layer_id2label


def map_layers_to_ids(layers: list[list[str]], layer_label2id: list[dict[str, int]]) -> tuple[list[list[int]], list[int], int]:
    labels: list[list[int]] = [[] for _ in range(len(layer_label2id))]
    label_mask: list[int] = [1] * len(layer_label2id)
    unknown = 0
    for level, names in enumerate(layers[: len(layer_label2id)]):
        level_unknown = 0
        for name in names:
            mapped = layer_label2id[level].get(name)
            if mapped is None:
                unknown += 1
                level_unknown += 1
                continue
            if mapped not in labels[level]:
                labels[level].append(mapped)
        if names and not labels[level] and level_unknown:
            label_mask[level] = 0
    return labels, label_mask, unknown


def restore_train_meta(meta: dict[str, Any]) -> tuple[list[dict[str, int]], list[dict[int, str]]]:
    layer_label2id = [{str(k): int(v) for k, v in level.items()} for level in meta["layer_label2id"]]
    layer_id2label = [{int(k): str(v) for k, v in level.items()} for level in meta["layer_id2label"]]
    return layer_label2id, layer_id2label


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_records(records: list[dict[str, Any]], num_levels: int) -> dict[str, Any]:
    valid_per_level = [0] * num_levels
    zero_label_valid_per_level = [0] * num_levels
    labels_per_level = [0] * num_levels
    for record in records:
        for level in range(num_levels):
            count = len(record["label"][level])
            labels_per_level[level] += count
            if record.get("label_mask", [1] * num_levels)[level]:
                valid_per_level[level] += 1
                if count == 0:
                    zero_label_valid_per_level[level] += 1
    return {
        "num_samples": len(records),
        "valid_samples_per_level": valid_per_level,
        "zero_label_valid_samples_per_level": zero_label_valid_per_level,
        "label_assignments_per_level": labels_per_level,
    }


def process_split(
    dataset: str,
    split: str,
    base_dir: Path,
    tree_path: Path | None,
    num_levels_arg: int | None,
    label_space_source: str,
) -> None:
    data_dir = base_dir / "data" / dataset
    if dataset == "rcv1" and split == "dev":
        input_path = data_dir / "rcv1_dev_ctr_com_0603.json"
    else:
        input_path = data_dir / f"{dataset}_{split}.json"
    output_path = data_dir / OUTPUT_FILES[split].format(dataset=dataset)
    meta_path = output_path.with_suffix(".meta.json")
    train_output_path = data_dir / OUTPUT_FILES["train"].format(dataset=dataset)
    train_meta_path = train_output_path.with_suffix(".meta.json")

    samples = load_jsonl(input_path)
    value_dict = load_value_dict(data_dir)
    tree_obj = load_tree(data_dir, dataset, tree_path)
    name_to_depth = build_tree_depth(tree_obj)
    inferred_num_levels = max(name_to_depth.values()) + 1
    configured_num_levels = num_levels_arg if num_levels_arg is not None else DEFAULT_NUM_LEVELS.get(dataset)
    num_levels = configured_num_levels or inferred_num_levels

    if num_levels > inferred_num_levels:
        raise ValueError(f"num_levels={num_levels} exceeds tree depth levels={inferred_num_levels}")

    layered: list[tuple[str, list[list[str]]]] = []
    raw_length_dist: Counter[int] = Counter()
    for sample in samples:
        text = infer_text(sample)
        label_groups = normalize_label_field(sample.get("label"))
        if not label_groups:
            continue
        if len(label_groups) == 1:
            raw_length_dist[len(label_groups[0])] += 1
        layers = labels_to_layers(label_groups, value_dict, name_to_depth, num_levels)
        if any(layers):
            layered.append((text, layers))

    if not layered:
        raise RuntimeError(f"No valid samples after tree mapping: {input_path}")

    if split == "train":
        if label_space_source == "train":
            layer_label2id, layer_id2label = build_label_space([layers for _, layers in layered], num_levels)
        elif label_space_source == "value_dict":
            full_label_names = value_dict_label_names(value_dict, name_to_depth, num_levels)
            if not full_label_names:
                raise RuntimeError("No valid labels found from value_dict and tree.")
            layer_label2id, layer_id2label = build_label_space_from_names(
                full_label_names,
                name_to_depth,
                num_levels,
            )
        else:
            raise ValueError(f"Unsupported label_space_source={label_space_source}")
    else:
        if not train_meta_path.exists():
            raise FileNotFoundError(f"Train meta not found: {train_meta_path}. Process train split first.")
        train_meta = load_json(train_meta_path)
        if int(train_meta["num_levels"]) != num_levels:
            raise ValueError(f"num_levels mismatch: train={train_meta['num_levels']} current={num_levels}")
        train_label_space_source = train_meta.get("label_space_source")
        if train_label_space_source is not None and train_label_space_source != label_space_source:
            raise ValueError(
                f"label_space_source mismatch: train={train_label_space_source} current={label_space_source}"
            )
        layer_label2id, layer_id2label = restore_train_meta(train_meta)

    records = []
    unknown_total = 0
    for text, layers in layered:
        labels, label_mask, unknown = map_layers_to_ids(layers, layer_label2id)
        unknown_total += unknown
        if any(label_mask):
            records.append({"query": text, "label": labels, "label_mask": label_mask})

    if not records:
        raise RuntimeError(f"No records kept after applying label space: {input_path}")

    write_jsonl(output_path, records)
    summary = summarize_records(records, num_levels)
    meta = {
        "dataset": dataset,
        "split": split,
        "mode": "flat_labels_with_tree_depth",
        "label_space_source": label_space_source,
        "num_levels": num_levels,
        "tree_num_levels": inferred_num_levels,
        "level_classes": [len(level) for level in layer_label2id],
        "total_classes": int(sum(len(level) for level in layer_label2id)),
        "layer_label2id": layer_label2id,
        "layer_id2label": layer_id2label,
        "raw_label_length_dist": dict(sorted(raw_length_dist.items())),
        "summary": summary,
        "unknown_labels_after_train_meta": unknown_total,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[{dataset}][{split}] input={input_path}")
    print(f"[{dataset}][{split}] output={output_path}")
    print(f"[{dataset}][{split}] samples_in={len(samples)} samples_out={len(records)}")
    print(f"[{dataset}][{split}] label_space_source={label_space_source}")
    print(f"[{dataset}][{split}] num_levels={num_levels} tree_num_levels={inferred_num_levels}")
    print(f"[{dataset}][{split}] level_classes={meta['level_classes']}")
    print(f"[{dataset}][{split}] valid_samples_per_level={summary['valid_samples_per_level']}")
    print(f"[{dataset}][{split}] unknown_labels_after_train_meta={unknown_total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess hierarchical labels into per-level multi-label format.")
    parser.add_argument("--dataset", choices=["wos", "rcv1", "nyt"])
    parser.add_argument("--split", choices=["train", "dev", "test"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--base-dir", default="/data1/lwq/agt_plus")
    parser.add_argument("--tree-path", default=None)
    parser.add_argument("--num-levels", type=int, default=None)
    parser.add_argument(
        "--label-space-source",
        choices=["value_dict", "train"],
        default="value_dict",
        help="Use value_dict+tree for a fixed full label space, or train for the old train-only label space.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    tree_path = Path(args.tree_path) if args.tree_path else None

    if args.all:
        datasets = ["wos", "rcv1", "nyt"] if args.dataset is None else [args.dataset]
        for dataset in datasets:
            for split in ["train", "dev", "test"]:
                process_split(dataset, split, base_dir, tree_path, args.num_levels, args.label_space_source)
    else:
        if args.dataset is None or args.split is None:
            raise ValueError("--dataset and --split are required unless --all is used")
        process_split(args.dataset, args.split, base_dir, tree_path, args.num_levels, args.label_space_source)


if __name__ == "__main__":
    main()
