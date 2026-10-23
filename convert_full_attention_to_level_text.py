"""Convert FP32 full-attention shards to a compact, lossless level-to-text bank.

CPU only: numpy and safetensors. No teacher/model/tokenizer imports. Read only
the level-token query rows; do not normalize, sort, select Top-k or quantize.
The original full bank is never modified. Use a fresh output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


FULL_FORMAT = "full_token_attention_head_mean_v1"
FORMAT = "level_text_attention_head_mean_v1"
MATCH_KEYS = ("dataset", "split", "teacher_checkpoint_fingerprint", "data_sha256",
              "tokenizer_sha256", "max_length", "truncate_mode", "num_levels",
              "level_classes", "level_offsets", "teacher_level_token_ids", "num_samples")
IDENTITY_KEYS = ("dataset_idx", "sample_uid", "source_line_no", "query_hash")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_fingerprint(manifest):
    content = {k: v for k, v in manifest.items() if k not in ("bank_fingerprint", "status", "error")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = handle.name
    os.replace(temporary, path)


def validate_full_manifest(manifest, layer):
    expected = {"format": FULL_FORMAT, "schema_version": 1, "status": "complete",
                "attention_distribution": "post_softmax_no_additional_normalization",
                "attention_head_reduction": "float32_mean", "causal_mask_preserved": True,
                "attention_scope": "all_valid_tokens_including_level_tokens_to_all_valid_tokens",
                "packing": "flat_row_major_squares_with_offsets", "storage_dtype": "float32"}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Source requires {key}={value!r}; got {manifest.get(key)!r}. FP16 cannot be restored to original FP32.")
    for key in (*MATCH_KEYS, "num_decoder_layers", "bank_fingerprint", "checkpoint_dir"):
        if key not in manifest:
            raise ValueError(f"Missing source metadata: {key}")
    total_layers = int(manifest["num_decoder_layers"])
    layer = layer if layer >= 0 else layer + total_layers
    if not 0 <= layer < total_layers or layer not in manifest["layer_ids"]:
        raise ValueError(f"Requested layer is not present: {layer}")
    if str(layer) not in manifest.get("attention_replay_validation", {}):
        raise ValueError("Source layer lacks attention replay validation")
    count, size = int(manifest["num_samples"]), int(manifest["shard_size"])
    levels = int(manifest["num_levels"])
    ids = manifest["teacher_level_token_ids"]
    if count < 1 or size < 1 or int(manifest["num_shards"]) != (count + size - 1) // size:
        raise ValueError("Invalid source sample/shard counts")
    if levels < 1 or len(ids) != levels or len(set(ids)) != levels or int(manifest["max_length"]) <= levels:
        raise ValueError("Invalid source level tokens/length")
    return layer


def iter_full_rows(root, manifest, layer, verify_checksums=False):
    """Stream one small H x T matrix plus token identities at a time."""
    root = Path(root)
    levels = int(manifest["num_levels"])
    level_ids = np.asarray(manifest["teacher_level_token_ids"], dtype=np.int64)
    count, size = int(manifest["num_samples"]), int(manifest["shard_size"])
    for shard in range(int(manifest["num_shards"])):
        start, end = shard * size, min(count, (shard + 1) * size)
        path = root / "full_attention" / f"layer-{layer:03d}" / f"shard-{shard:05d}.safetensors"
        sidecar = read_json(path.with_suffix(".json"))
        expected = {"format": FULL_FORMAT, "bank_fingerprint": manifest["bank_fingerprint"],
                    "layer": layer, "start_idx": start, "end_idx": end}
        if any(sidecar.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Source shard metadata mismatch: {path}")
        if verify_checksums and sidecar.get("sha256") != sha256_file(path):
            raise ValueError(f"Source shard checksum mismatch: {path}")
        with safe_open(str(path), framework="np", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if any(metadata.get(k) != expected[k] for k in ("format", "bank_fingerprint")):
                raise ValueError(f"Source tensor metadata mismatch: {path}")
            identities = {k: handle.get_tensor(k) for k in IDENTITY_KEYS}
            for key, shape, dtype in (("dataset_idx", (end-start,), np.int64),
                                      ("source_line_no", (end-start,), np.int64),
                                      ("sample_uid", (end-start, 32), np.uint8),
                                      ("query_hash", (end-start, 32), np.uint8)):
                if identities[key].shape != shape or identities[key].dtype != dtype:
                    raise ValueError(f"Invalid source identity tensor: {key}")
            if not np.array_equal(identities["dataset_idx"], np.arange(start, end)):
                raise ValueError("Source sample order mismatch")
            if (identities["source_line_no"] < 1).any():
                raise ValueError("Invalid source line numbers")
            lengths = handle.get_tensor("valid_lengths")
            if (lengths.shape != (end-start,) or lengths.dtype != np.int64
                    or (lengths <= levels).any() or (lengths > manifest["max_length"]).any()):
                raise ValueError("Invalid source sequence lengths")
            tokens_offset, matrices_offset = handle.get_tensor("token_offsets"), handle.get_tensor("matrix_offsets")
            for offsets, sizes in ((tokens_offset, lengths), (matrices_offset, lengths ** 2)):
                if offsets.dtype != np.int64 or not np.array_equal(offsets, np.r_[0, np.cumsum(sizes)]):
                    raise ValueError("Invalid source offsets")
            tokens, matrices = handle.get_slice("input_ids"), handle.get_slice("teacher_full_attention_mean")
            if tokens.get_shape() != [int(tokens_offset[-1])] or matrices.get_shape() != [int(matrices_offset[-1])]:
                raise ValueError("Source tensor sizes differ from offsets")
            for row, n_value in enumerate(lengths):
                n, t = int(n_value), int(n_value) - levels
                ids = tokens[int(tokens_offset[row]):int(tokens_offset[row + 1])]
                if (ids.dtype != np.int64 or not np.array_equal(ids[t:], level_ids)
                        or np.isin(ids[:t], level_ids).any() or (ids < 0).any()):
                    raise ValueError(f"Source token layout mismatch at sample {start + row}")
                begin = int(matrices_offset[row]) + t * n
                values = matrices[begin:int(matrices_offset[row + 1])].reshape(levels, n)
                if values.dtype != np.float32 or not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("Invalid FP32 source attention rows")
                if not np.allclose(values.sum(-1, dtype=np.float64), 1, atol=0.02, rtol=0.02):
                    raise ValueError("Source attention rows do not sum to one")
                if any((values[q, t + q + 1:] > 1e-5).any() for q in range(levels)):
                    raise ValueError("Source level row attends to a future token")
                raw = np.ascontiguousarray(values[:, :t])
                if (raw.sum(-1, dtype=np.float64) <= 0).any():
                    raise ValueError("Nonpositive source text attention mass")
                yield {**{k: v[row].copy() for k, v in identities.items()},
                       "input_ids": ids.copy(), "raw_attention": raw}


def pack_samples(samples):
    lengths = np.array([len(s["input_ids"]) for s in samples], dtype=np.int64)
    text_lengths = np.array([s["raw_attention"].shape[1] for s in samples], dtype=np.int64)
    sizes = np.array([s["raw_attention"].size for s in samples], dtype=np.int64)
    return {**{k: np.stack([s[k] for s in samples]) for k in IDENTITY_KEYS},
            "valid_lengths": lengths, "text_lengths": text_lengths,
            "token_offsets": np.r_[np.int64(0), np.cumsum(lengths)],
            "row_offsets": np.r_[np.int64(0), np.cumsum(sizes)],
            "input_ids": np.concatenate([s["input_ids"] for s in samples]),
            "teacher_text_attention_raw_mean": np.concatenate([s["raw_attention"].reshape(-1) for s in samples])}


def read_light_manifest(root):
    m = read_json(Path(root) / "manifest.json")
    expected = {"format": FORMAT, "schema_version": 1, "status": "complete", "storage_dtype": "float32",
                "source_storage_dtype": "float32", "attention_head_reduction": "float32_mean",
                "attention_distribution": "post_softmax_no_additional_normalization",
                "attention_scope": "level_token_to_all_text_tokens", "packing": "flat_level_text_rows_with_offsets"}
    if any(m.get(k) != v for k, v in expected.items()):
        raise ValueError("Expected a complete FP32 compact level-to-text bank")
    if m.get("bank_fingerprint") != manifest_fingerprint(m):
        raise ValueError("Compact bank manifest fingerprint mismatch")
    count, size, levels = int(m["num_samples"]), int(m["shard_size"]), int(m["num_levels"])
    if count < 1 or size < 1 or m["num_shards"] != (count + size - 1) // size:
        raise ValueError("Invalid compact bank counts")
    if (levels < 1 or len(m["teacher_level_token_ids"]) != levels or len(set(m["teacher_level_token_ids"])) != levels
            or not 0 <= m["teacher_layer"] < m["num_decoder_layers"]
            or str(m["teacher_layer"]) not in m.get("attention_replay_validation", {})):
        raise ValueError("Invalid compact bank layer/level metadata")
    return m


def read_light_shard(root, manifest, shard):
    """Validate checksum/schema and return only one small shard as numpy arrays."""
    start = shard * int(manifest["shard_size"])
    end = min(int(manifest["num_samples"]), start + int(manifest["shard_size"]))
    if start < 0 or start >= end:
        raise IndexError(shard)
    path = Path(root) / "attention" / f"shard-{shard:05d}.safetensors"
    sidecar = read_json(path.with_suffix(".json"))
    metadata = {"format": FORMAT, "bank_fingerprint": manifest["bank_fingerprint"]}
    if (any(sidecar.get(k) != v for k, v in {**metadata, "start_idx": start, "end_idx": end}.items())
            or sidecar.get("sha256") != sha256_file(path)):
        raise ValueError(f"Compact shard checksum/metadata mismatch: {path}")
    with safe_open(str(path), framework="np", device="cpu") as handle:
        if any((handle.metadata() or {}).get(k) != v for k, v in metadata.items()):
            raise ValueError("Compact tensor metadata mismatch")
        values = {k: handle.get_tensor(k) for k in handle.keys()}
    required = {*IDENTITY_KEYS, "valid_lengths", "text_lengths", "token_offsets", "row_offsets",
                "input_ids", "teacher_text_attention_raw_mean"}
    if set(values) != required:
        raise ValueError("Compact tensor fields mismatch")
    for k in required:
        dtype = np.uint8 if k in ("sample_uid", "query_hash") else np.float32 if k == "teacher_text_attention_raw_mean" else np.int64
        if values[k].dtype != dtype:
            raise ValueError(f"Invalid compact tensor dtype: {k}")
    for k in (*IDENTITY_KEYS, "valid_lengths", "text_lengths"):
        shape = (end-start, 32) if k in ("sample_uid", "query_hash") else (end-start,)
        if values[k].shape != shape:
            raise ValueError(f"Invalid compact tensor shape: {k}")
    if not np.array_equal(values["dataset_idx"], np.arange(start, end)) or (values["source_line_no"] < 1).any():
        raise ValueError("Compact sample order/line numbers mismatch")
    levels, lengths = int(manifest["num_levels"]), values["text_lengths"]
    if (lengths < 1).any() or (lengths + levels > manifest["max_length"]).any() or not np.array_equal(values["valid_lengths"], lengths + levels):
        raise ValueError("Invalid compact text lengths")
    for name, sizes in (("token_offsets", lengths + levels), ("row_offsets", lengths * levels)):
        if not np.array_equal(values[name], np.r_[0, np.cumsum(sizes)]):
            raise ValueError(f"Invalid compact offsets: {name}")
    if values["input_ids"].shape != (int(values["token_offsets"][-1]),) or values["teacher_text_attention_raw_mean"].shape != (int(values["row_offsets"][-1]),):
        raise ValueError("Compact flattened sizes mismatch")
    for i, length in enumerate(lengths):
        ids = values["input_ids"][values["token_offsets"][i]:values["token_offsets"][i+1]]
        if (not np.array_equal(ids[length:], manifest["teacher_level_token_ids"])
                or np.isin(ids[:length], manifest["teacher_level_token_ids"]).any() or (ids < 0).any()):
            raise ValueError("Compact token layout mismatch")
        raw = values["teacher_text_attention_raw_mean"][values["row_offsets"][i]:values["row_offsets"][i+1]].reshape(levels, int(length))
        mass = raw.sum(-1, dtype=np.float64)
        if not np.isfinite(raw).all() or (raw < 0).any() or (mass <= 0).any() or (mass > 1.04).any():
            raise ValueError("Invalid compact text attention")
    return values


def convert(full_attention_dir, output_dir, layer=-1, shard_size=256, verify_source_checksums=False):
    source, output = Path(full_attention_dir).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use separate, non-nested source and output directories")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output is not empty; use a new directory (no files are overwritten)")
    full = read_json(source / "manifest.json")
    layer = validate_full_manifest(full, layer)
    m = {k: full[k] for k in MATCH_KEYS}
    m.update(format=FORMAT, schema_version=1, source_full_attention_dir=str(source),
             source_bank_fingerprint=full["bank_fingerprint"], source_storage_dtype=full["storage_dtype"],
             checkpoint_dir=full["checkpoint_dir"], num_decoder_layers=full["num_decoder_layers"],
             teacher_layer=layer, source_num_samples=full.get("source_num_samples", full["num_samples"]),
             storage_dtype="float32", attention_head_reduction="float32_mean",
             attention_distribution="post_softmax_no_additional_normalization",
             attention_scope="level_token_to_all_text_tokens", packing="flat_level_text_rows_with_offsets",
             attention_replay_validation={str(layer): full["attention_replay_validation"][str(layer)]},
             shard_size=shard_size, num_shards=(full["num_samples"] + shard_size - 1) // shard_size,
             source_file_checksums_verified=bool(verify_source_checksums),
             converter_script_sha256=sha256_file(__file__),
             conversion="slice_FP32_level_to_text_rows_without_normalization_sorting_or_rounding")
    m["bank_fingerprint"] = manifest_fingerprint(m)
    output.mkdir(parents=True, exist_ok=True)
    (output / "attention").mkdir()
    atomic_json(output / "manifest.json", {**m, "status": "converting"})
    pending, written, shard = [], 0, 0

    def flush(samples, number, start):
        payload = pack_samples(samples)
        path = output / "attention" / f"shard-{number:05d}.safetensors"
        temporary = path.with_suffix(".safetensors.tmp")
        metadata = {"format": FORMAT, "bank_fingerprint": m["bank_fingerprint"]}
        save_file(payload, str(temporary), metadata=metadata)
        os.replace(temporary, path)
        atomic_json(path.with_suffix(".json"), {**metadata, "start_idx": start,
                    "end_idx": start + len(samples), "sha256": sha256_file(path)})
        # Validate our output before publishing completion.
        read_light_shard(output, m, number)

    try:
        for sample in iter_full_rows(source, full, layer, verify_source_checksums):
            pending.append(sample)
            if len(pending) == shard_size:
                flush(pending, shard, written)
                written += len(pending)
                shard += 1
                pending.clear()
                print(f"Converted {written}/{m['num_samples']} samples", flush=True)
        if pending:
            flush(pending, shard, written)
            written += len(pending)
            shard += 1
        if written != m["num_samples"] or shard != m["num_shards"]:
            raise ValueError("Incomplete compact bank")
        atomic_json(output / "manifest.json", {**m, "status": "complete"})
    except Exception as exc:
        atomic_json(output / "manifest.json", {**m, "status": "incomplete", "error": str(exc)})
        raise
    print(f"Compact attention complete: {output}; samples={written}, layer={layer}, dtype=float32")
    return read_light_manifest(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full_attention_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layer", type=int, default=-1, help="Zero-based layer; -1 selects the final Transformer layer")
    parser.add_argument("--shard_size", type=int, default=256)
    parser.add_argument("--verify_source_checksums", action="store_true",
                        help="Also stream/hash entire source files; more disk I/O, no full-matrix RAM allocation")
    args = parser.parse_args()
    try:
        convert(**vars(args))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Conversion failed: {exc}\n")


if __name__ == "__main__":
    main()
