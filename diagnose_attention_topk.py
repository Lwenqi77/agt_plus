"""CPU diagnostics for full_token_attention_head_mean_v1 caches.

Only level-token query rows are read. No model, tokenizer, or training imports.
Dependencies: numpy, safetensors. See ATTENTION_TOPK_DIAGNOSTICS.md.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


FORMAT = "full_token_attention_head_mean_v1"
KS = (16, 32, 64, 128, 256, 512, 1024, 2048)
PERCENTAGES = (90, 95, 99)
METRICS = ("text_mass", *(f"C{k}" for k in KS), "C_all", *(f"K{p}" for p in PERCENTAGES))
LENGTH_BINS = ((1, 64), (65, 128), (129, 256), (257, 512), (513, 1024),
               (1025, 2048), (2049, None))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        raise ValueError("Cannot summarize an empty population")
    quantiles = np.percentile(values, [5, 10, 25, 50, 75, 90, 95, 99])
    return {"count": int(values.size), "mean": float(values.mean()),
            "min": float(values.min()), "max": float(values.max()),
            **dict(zip(("p05", "p10", "p25", "median", "p75", "p90", "p95", "p99"),
                       map(float, quantiles)))}


def normalize_topks(topks):
    values = tuple(topks)
    if not values or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in values):
        raise ValueError("topks must be nonempty positive integers")
    # Preserve C32/C64 and existing reports consumed by earlier tools.
    return tuple(sorted(set(values) | {32, 64}))


def parse_topks(value):
    try:
        return normalize_topks(tuple(int(k.strip()) for k in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def row_metrics(text_rows, topks=KS):
    """Normalize each label-level row within text, then sort tokens.

    Head averaging has already happened in the extractor. Accumulate stored
    FP16/FP32 values in float64 for diagnostics; this cannot undo quantization.
    """
    scores = np.asarray(text_rows, dtype=np.float64)
    if (scores.ndim != 2 or scores.shape[1] == 0
            or not np.isfinite(scores).all() or (scores < 0).any()):
        raise ValueError("Invalid level-to-text attention")
    mass = scores.sum(axis=1)
    if (mass <= 0).any():
        raise ValueError("Nonpositive text attention mass")
    ordered = np.sort(scores / mass[:, None], axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1)
    cumulative[:, -1] = 1.0
    result = {"text_mass": mass}
    for k in normalize_topks(topks):
        result[f"C{k}"] = cumulative[:, min(k, scores.shape[1]) - 1]
    result["C_all"] = cumulative[:, -1]
    for percentage in PERCENTAGES:
        # Small tolerance avoids float64 boundary noise, e.g. exactly 0.9.
        result[f"K{percentage}"] = (cumulative < percentage / 100 - 1e-12).sum(axis=1) + 1
    return result


def compare_reference(manifest, reference_path, layer):
    if reference_path is None:
        return None
    reference = read_json(reference_path)
    for key in ("dataset", "split", "teacher_checkpoint_fingerprint", "data_sha256",
                "tokenizer_sha256", "max_length", "truncate_mode", "num_levels",
                "level_classes", "level_offsets", "teacher_level_token_ids"):
        if key not in reference or key not in manifest or reference[key] != manifest[key]:
            raise ValueError(f"Reference manifest mismatch or missing field: {key}")
    ref_layer = reference.get("attention_layer_resolved", reference.get("attention_layer"))
    if ref_layer is None:
        raise ValueError("Reference has no single attention_layer; fused-layer targets require a separate diagnostic")
    ref_layer = int(ref_layer)
    if ref_layer < 0:
        ref_layer += int(manifest["num_decoder_layers"])
    if ref_layer != layer:
        raise ValueError(f"Reference attention layer {ref_layer} differs from selected layer {layer}")
    return {"path": str(Path(reference_path).resolve()), "metadata_and_layer_match": True,
            "attention_topk": reference.get("attention_topk"),
            "note": "Metadata comparison only; no per-sample comparison to the reference cache."}


def inspect_cache(root, layer=-1, max_samples=None, reference_manifest=None, topks=KS):
    topks = normalize_topks(topks)
    metric_names = ("text_mass", *(f"C{k}" for k in topks), "C_all", *(f"K{p}" for p in PERCENTAGES))
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("format") != FORMAT or manifest.get("status") != "complete":
        raise ValueError("Expected a complete full_token_attention_head_mean_v1 cache")
    if (manifest.get("attention_head_reduction") != "float32_mean"
            or manifest.get("packing") != "flat_row_major_squares_with_offsets"
            or manifest.get("storage_dtype") not in ("float16", "float32")):
        raise ValueError("Unsupported attention reduction, packing, or storage dtype")
    if layer < 0:
        layer += int(manifest["num_decoder_layers"])
    if layer not in manifest["layer_ids"]:
        raise ValueError(f"Layer {layer} not saved; available layers: {manifest['layer_ids']}")
    levels = int(manifest["num_levels"])
    level_ids = np.asarray(manifest["teacher_level_token_ids"], dtype=np.int64)
    count, shard_size = int(manifest["num_samples"]), int(manifest["shard_size"])
    if levels < 1 or len(level_ids) != levels or len(set(level_ids.tolist())) != levels:
        raise ValueError("Invalid level token metadata")
    if count < 1 or shard_size < 1 or (max_samples is not None and max_samples < 1):
        raise ValueError("Sample and shard counts must be positive")
    if int(manifest["num_shards"]) != (count + shard_size - 1) // shard_size:
        raise ValueError("Inconsistent shard count in manifest")
    reference = compare_reference(manifest, reference_manifest, layer)
    limit = min(count, max_samples) if max_samples is not None else count
    lengths_out = np.empty(limit, dtype=np.int64)
    metrics = {key: np.empty((limit, levels), dtype=np.float64) for key in metric_names}
    shards_to_read = (limit + shard_size - 1) // shard_size
    for shard in range(shards_to_read):
        start, end = shard * shard_size, min(count, (shard + 1) * shard_size)
        path = root / "full_attention" / f"layer-{layer:03d}" / f"shard-{shard:05d}.safetensors"
        sidecar = read_json(path.with_suffix(".json"))
        expected = {"format": FORMAT, "bank_fingerprint": manifest["bank_fingerprint"],
                    "layer": layer, "start_idx": start, "end_idx": end}
        if any(sidecar.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Shard metadata mismatch: {path}")
        with safe_open(str(path), framework="np", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if any(metadata.get(k) != expected[k] for k in ("format", "bank_fingerprint")):
                raise ValueError(f"Tensor metadata mismatch: {path}")
            indices = handle.get_tensor("dataset_idx")
            lengths = handle.get_tensor("valid_lengths")
            token_offsets = handle.get_tensor("token_offsets")
            matrix_offsets = handle.get_tensor("matrix_offsets")
            if indices.dtype != np.int64 or not np.array_equal(indices, np.arange(start, end)):
                raise ValueError(f"Missing, duplicate, or out-of-order sample indices: {path}")
            if (lengths.dtype != np.int64 or lengths.shape != (end - start,)
                    or (lengths <= levels).any() or (lengths > manifest["max_length"]).any()):
                raise ValueError(f"Invalid lengths: {path}")
            for offsets, sizes in ((token_offsets, lengths), (matrix_offsets, lengths ** 2)):
                if offsets.dtype != np.int64 or not np.array_equal(offsets, np.r_[0, np.cumsum(sizes)]):
                    raise ValueError(f"Invalid ragged offsets: {path}")
            tokens = handle.get_slice("input_ids")
            matrices = handle.get_slice("teacher_full_attention_mean")
            if (tokens.get_shape() != [int(token_offsets[-1])]
                    or matrices.get_shape() != [int(matrix_offsets[-1])]):
                raise ValueError(f"Tensor sizes disagree with offsets: {path}")
            for i in range(min(end, limit) - start):
                n = int(lengths[i])
                text_length = n - levels
                ids = tokens[int(token_offsets[i]):int(token_offsets[i + 1])]
                if (ids.dtype != np.int64 or not np.array_equal(ids[text_length:], level_ids)
                        or np.isin(ids[:text_length], level_ids).any()):
                    raise ValueError(f"Level tokens are not a unique ordered suffix: sample {start + i}")
                # The last `levels` rows are contiguous in the flattened matrix.
                begin = int(matrix_offsets[i]) + text_length * n
                rows = matrices[begin:int(matrix_offsets[i + 1])].reshape(levels, n)
                if rows.dtype != np.dtype(manifest["storage_dtype"]):
                    raise ValueError(f"Storage dtype mismatch: {path}")
                if (not np.isfinite(rows).all() or (rows < 0).any()
                        or not np.allclose(rows.sum(axis=1, dtype=np.float64), 1, atol=0.02, rtol=0.02)):
                    raise ValueError(f"Invalid selected attention rows: sample {start + i}")
                values = row_metrics(rows[:, :text_length], topks)
                lengths_out[start + i] = text_length
                for key in metric_names:
                    metrics[key][start + i] = values[key]
        if shard == 0 or (shard + 1) % 100 == 0 or shard + 1 == shards_to_read:
            print(f"Read {shard + 1}/{shards_to_read} shards; {min(end, limit)}/{limit} samples", flush=True)
    provenance_keys = ("dataset", "split", "checkpoint_dir", "teacher_checkpoint_fingerprint",
                       "data_sha256", "tokenizer_sha256", "max_length", "truncate_mode",
                       "num_levels", "teacher_level_token_ids", "num_decoder_layers",
                       "storage_dtype", "bank_fingerprint")
    report = {"knowledge_dir": str(root), "metadata": {k: manifest.get(k) for k in provenance_keys},
              "layer": layer, "analyzed_samples": limit, "cache_samples": count,
              "source_samples": manifest.get("source_num_samples", count),
              "prefix_only": limit < int(manifest.get("source_num_samples", count)),
              "reference_check": reference,
              "topks": list(topks), "metric_names": list(metric_names),
              "coverage_population": "All saved sample/label-level rows, equally weighted; label_mask is not applied.",
              "definition": "Head mean already stored; select level-to-text rows, normalize within text, sort per sample and label level.",
              "validation": "Checks metadata, indices, offsets, token suffix, and selected row probabilities. Does not hash whole shard files or inspect other query rows.",
              "length": summarize(lengths_out),
              "r32": summarize(np.minimum(32, lengths_out) / lengths_out),
              "fraction_at_text_length_cap": float(np.mean(lengths_out == int(manifest["max_length"]) - levels)),
              "levels": {}}
    for level in range(levels):
        report["levels"][str(level + 1)] = {
            **{key: summarize(metrics[key][:, level]) for key in metric_names},
            **{f"fraction_C{k}_at_least_0_99": float(np.mean(metrics[f"C{k}"][:, level] >= 0.99 - 1e-12))
               for k in topks},
            "fraction_C32_below_0_90": float(np.mean(metrics["C32"][:, level] < 0.90 - 1e-12)),
            "fraction_C32_below_0_80": float(np.mean(metrics["C32"][:, level] < 0.80 - 1e-12)),
            "mean_C64_minus_C32": float(np.mean(metrics["C64"][:, level] - metrics["C32"][:, level])),
        }
    report["overall"] = {key: summarize(value.reshape(-1)) for key, value in metrics.items()}
    report["coverage_by_k"] = []
    for k in (*topks, "all"):
        coverage = metrics[f"C{k}" if k != "all" else "C_all"]
        reached = coverage >= 0.99 - 1e-12
        report["coverage_by_k"].append({
            "k": k, "mean_coverage": float(coverage.mean()), "min_coverage": float(coverage.min()),
            "p10_coverage": float(np.percentile(coverage, 10)),
            "fraction_rows_at_least_99": float(reached.mean()),
            "fraction_samples_all_levels_at_least_99": float(reached.all(axis=1).mean()),
        })
    report["fixed_k_for_99_all_analyzed_rows"] = int(metrics["K99"].max())
    report["smallest_tested_k_for_mean_99"] = next(
        (r["k"] for r in report["coverage_by_k"] if r["k"] != "all" and r["mean_coverage"] >= 0.99 - 1e-12), None)
    return report, lengths_out, metrics


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output_dir, report, lengths, metrics, save_samples=False):
    metric_names = report.get("metric_names", list(metrics))
    output = Path(output_dir).resolve()
    source = Path(report["knowledge_dir"]).resolve()
    if output == source or source in output.parents:
        raise ValueError("Use an output directory outside the teacher cache")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    summary = [{"label_level": "all", "metric": key, **report[key]}
               for key in ("length", "r32")]
    for level, values in report["levels"].items():
        summary.extend({"label_level": level, "metric": key, **values[key]} for key in metric_names)
    summary.extend({"label_level": "all_rows", "metric": key, **value}
                   for key, value in report.get("overall", {}).items())
    write_csv(output / "summary.csv", summary)
    if report.get("coverage_by_k"):
        write_csv(output / "coverage_by_k.csv", report["coverage_by_k"])
    bins = []
    for lower, upper in LENGTH_BINS:
        selected = lengths >= lower
        if upper is not None:
            selected &= lengths <= upper
        if not selected.any():
            continue
        for level in range(metrics["C32"].shape[1]):
            coverage = metrics["C32"][selected, level]
            bins.append({"length_bin": f"{lower}-{upper}" if upper is not None else f"{lower}+",
                         "label_level": level + 1, "count": int(selected.sum()),
                         "mean_length": float(lengths[selected].mean()),
                         "mean_r32": float((np.minimum(32, lengths[selected]) / lengths[selected]).mean()),
                         **{f"mean_{key}": float(metrics[key][selected, level].mean()) for key in metric_names},
                         **{f"fraction_C{k}_at_least_0_99": float(np.mean(metrics[f"C{k}"][selected, level] >= 0.99 - 1e-12))
                            for k in report.get("topks", KS)},
                         "p10_C32": float(np.percentile(coverage, 10)),
                         "median_C32": float(np.median(coverage)),
                         "fraction_C32_below_0_90": float(np.mean(coverage < 0.90 - 1e-12))})
    write_csv(output / "length_bins.csv", bins)
    if save_samples:
        with (output / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dataset_idx", "label_level", "text_length", "r32", *metric_names])
            for i, length in enumerate(lengths):
                for level in range(metrics["C32"].shape[1]):
                    writer.writerow([i, level + 1, int(length), min(32, length) / length,
                                     *[float(metrics[key][i, level]) for key in metric_names]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge_dir", required=True)
    parser.add_argument("--output_dir", required=True, help="Separate report directory outside the cache")
    parser.add_argument("--layer", type=int, default=-1, help="Zero-based Transformer layer; -1 means model's last layer")
    parser.add_argument("--max_samples", type=int, help="First N samples for smoke testing only; default scans all")
    parser.add_argument("--reference_manifest", help="Optional original top-k KD manifest for metadata and layer checks")
    parser.add_argument("--save_samples", action="store_true", help="Also write one CSV row per sample and label level")
    parser.add_argument("--topks", type=parse_topks, default=KS,
                        help="Comma-separated positive k values; default 16,32,64,128,256,512,1024,2048; C32/C64 and All always included")
    args = parser.parse_args()
    try:
        report, lengths, metrics = inspect_cache(args.knowledge_dir, args.layer, args.max_samples, args.reference_manifest, args.topks)
        write_reports(args.output_dir, report, lengths, metrics, args.save_samples)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Diagnostic failed: {exc}\n")
    print(f"Dataset={report['metadata']['dataset']} Transformer layer={report['layer']} samples={report['analyzed_samples']} prefix_only={report['prefix_only']}")
    print(f"Text length: mean={report['length']['mean']:.2f}, median={report['length']['median']:.2f}, P95={report['length']['p95']:.2f}")
    print("K | mean coverage | rows reaching 99% | samples with ALL levels reaching 99%")
    for row in report["coverage_by_k"]:
        print(f"{str(row['k']):>4} | {row['mean_coverage']:.6f} | {row['fraction_rows_at_least_99']:.6f} | {row['fraction_samples_all_levels_at_least_99']:.6f}")
    print("Label level | K90 median | K95 median | K99 median | K99 P95 | K99 max")
    for level, values in report["levels"].items():
        print(f"{level:>11} | {values['K90']['median']:.1f} | {values['K95']['median']:.1f} | {values['K99']['median']:.1f} | {values['K99']['p95']:.1f} | {values['K99']['max']:.0f}")
    print(f"Fixed k covering 99% in every analyzed row: {report['fixed_k_for_99_all_analyzed_rows']}")
    print(f"Reports: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
