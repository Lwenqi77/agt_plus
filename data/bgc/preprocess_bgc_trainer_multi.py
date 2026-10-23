#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def norm_name(x: Any) -> str:
    return unicodedata.normalize("NFC", str(x).strip())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSON: {path}:{line_no}: {exc}"
                ) from exc

    if not records:
        raise RuntimeError(f"No samples found: {path}")

    return records


def load_taxonomy(
    taxonomy_path: Path,
) -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
]:
    children: dict[str, set[str]] = defaultdict(set)
    parents: dict[str, set[str]] = defaultdict(set)

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")

            if not line.strip():
                continue

            parts = [
                norm_name(x)
                for x in line.split("\t")
                if norm_name(x)
            ]

            if not parts:
                continue

            parent = parts[0]

            for child in parts[1:]:
                children[parent].add(child)
                parents[child].add(parent)

    if "Root" not in children:
        raise RuntimeError(
            "Taxonomy does not contain a Root entry."
        )

    return children, parents


def build_depths(
    children: dict[str, set[str]],
) -> dict[str, int]:
    """
    Root itself is excluded.

    Root's children -> level/depth 0
    grandchildren  -> level/depth 1
    ...
    """
    depth: dict[str, int] = {}

    queue = deque()

    for child in sorted(children["Root"]):
        depth[child] = 0
        queue.append(child)

    while queue:
        parent = queue.popleft()
        parent_depth = depth[parent]

        for child in sorted(children.get(parent, set())):
            new_depth = parent_depth + 1

            if child not in depth:
                depth[child] = new_depth
                queue.append(child)

            elif depth[child] != new_depth:
                raise RuntimeError(
                    f"Label appears at inconsistent taxonomy depths: "
                    f"{child!r}: "
                    f"{depth[child]} vs {new_depth}"
                )

    if not depth:
        raise RuntimeError("No labels found below Root.")

    return depth


def build_label_space(
    name_to_depth: dict[str, int],
    num_levels: int,
) -> tuple[
    list[dict[str, int]],
    list[dict[int, str]],
]:
    names_by_level: list[list[str]] = [
        [] for _ in range(num_levels)
    ]

    for name, depth in name_to_depth.items():
        if depth < num_levels:
            names_by_level[depth].append(name)

    layer_label2id = []
    layer_id2label = []

    for names in names_by_level:
        names = sorted(set(names))

        label2id = {
            name: idx
            for idx, name in enumerate(names)
        }

        id2label = {
            idx: name
            for name, idx in label2id.items()
        }

        layer_label2id.append(label2id)
        layer_id2label.append(id2label)

    return layer_label2id, layer_id2label


def convert_split(
    input_path: Path,
    output_path: Path,
    meta_path: Path,
    name_to_depth: dict[str, int],
    layer_label2id: list[dict[str, int]],
) -> None:
    samples = load_jsonl(input_path)

    num_levels = len(layer_label2id)

    records = []

    unknown_labels = set()

    label_assignments_per_level = [0] * num_levels
    zero_label_samples_per_level = [0] * num_levels

    for sample_idx, sample in enumerate(samples):
        if "token" not in sample:
            raise KeyError(
                f"Sample {sample_idx} has no 'token': "
                f"{sample.keys()}"
            )

        text = str(sample["token"])

        raw_labels = sample.get("label")

        if not isinstance(raw_labels, list):
            raise TypeError(
                f"Sample {sample_idx} label is not a list: "
                f"{raw_labels!r}"
            )

        labels: list[list[int]] = [
            [] for _ in range(num_levels)
        ]

        for raw_name in raw_labels:
            name = norm_name(raw_name)

            depth = name_to_depth.get(name)

            if depth is None:
                unknown_labels.add(name)
                continue

            mapped = layer_label2id[depth].get(name)

            if mapped is None:
                unknown_labels.add(name)
                continue

            if mapped not in labels[depth]:
                labels[depth].append(mapped)

        for level in range(num_levels):
            labels[level].sort()

            label_assignments_per_level[level] += len(
                labels[level]
            )

            if not labels[level]:
                zero_label_samples_per_level[level] += 1

        # BGC 原始任务把未出现的标签视为负标签，
        # 因此所有 level 均参与 BCE。
        label_mask = [1] * num_levels

        records.append(
            {
                "query": text,
                "label": labels,
                "label_mask": label_mask,
            }
        )

    if unknown_labels:
        examples = sorted(unknown_labels)[:30]

        raise RuntimeError(
            "Some labels in the dataset are missing from taxonomy. "
            f"count={len(unknown_labels)}, "
            f"examples={examples}"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

    meta = {
        "dataset": "bgc",
        "split": output_path.name.split("_")[1],
        "mode": "flat_labels_with_tree_depth",
        "label_space_source": "taxonomy",
        "num_levels": num_levels,
        "tree_num_levels": num_levels,
        "level_classes": [
            len(x)
            for x in layer_label2id
        ],
        "total_classes": sum(
            len(x)
            for x in layer_label2id
        ),
        "layer_label2id": layer_label2id,
        "layer_id2label": [
            {
                str(k): v
                for k, v in level.items()
            }
            for level in layer_id2label_global
        ],
        "summary": {
            "num_samples": len(records),
            "valid_samples_per_level": [
                len(records)
            ] * num_levels,
            "zero_label_valid_samples_per_level":
                zero_label_samples_per_level,
            "label_assignments_per_level":
                label_assignments_per_level,
        },
        "unknown_labels_after_taxonomy": 0,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            meta,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"input  = {input_path}")
    print(f"output = {output_path}")
    print(f"samples = {len(records)}")
    print(
        "zero_label_valid_samples_per_level = "
        f"{zero_label_samples_per_level}"
    )
    print(
        "label_assignments_per_level = "
        f"{label_assignments_per_level}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="/data1/lwq/agt_plus/data/BGC",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    taxonomy_path = data_dir / "bgc.taxonomy"

    children, parents = load_taxonomy(
        taxonomy_path
    )

    name_to_depth = build_depths(children)

    num_levels = max(name_to_depth.values()) + 1

    print(f"taxonomy labels = {len(name_to_depth)}")
    print(f"num_levels = {num_levels}")

    for level in range(num_levels):
        names = sorted(
            name
            for name, depth
            in name_to_depth.items()
            if depth == level
        )

        print(
            f"level {level + 1}: "
            f"{len(names)} labels"
        )

    layer_label2id_global, layer_id2label_global = (
        build_label_space(
            name_to_depth,
            num_levels,
        )
    )

    for split, source_name in [
        ("train", "train_data.jsonl"),
        ("dev", "dev_data.jsonl"),
        ("test", "test_data.jsonl"),
    ]:
        convert_split(
            input_path=data_dir / source_name,
            output_path=(
                data_dir
                / f"bgc_{split}_trainer_multi.json"
            ),
            meta_path=(
                data_dir
                / f"bgc_{split}_trainer_multi.meta.json"
            ),
            name_to_depth=name_to_depth,
            layer_label2id=layer_label2id_global,
        )