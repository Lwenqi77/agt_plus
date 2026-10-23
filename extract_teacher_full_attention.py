"""Extract post-softmax, head-mean attention over all valid token pairs.

Independent ragged full-attention format; does not modify the existing bank.
See FULL_ATTENTION_EXTRACTION.md. Only inference is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from extract_teacher_multilayer import old, resolve_layers, synchronized_error

FORMAT = "full_token_attention_head_mean_v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wos", choices=sorted(old.DATASET_ZOO))
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--layers", default="last:1", help="all | last:K | comma-separated zero-based indices")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shard_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, help="Prefix smoke test; use a separate output directory")
    parser.add_argument("--storage_dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--overwrite_shards", action="store_true")
    return parser.parse_args()


def full_attention_rows(attention, mask):
    """FP32 head mean, then remove right padding; retain causal zeros and level tokens."""
    if attention.ndim != 4 or attention.shape[0] != mask.shape[0]:
        raise ValueError("Expected attention [batch, heads, seq, seq]")
    if attention.shape[-2:] != (mask.shape[1], mask.shape[1]):
        raise ValueError("Attention and token lengths differ")
    rows = []
    for b in range(mask.shape[0]):
        length = int(mask[b].sum().item())
        if length <= 0 or not bool((mask[b, :length] == 1).all()) or bool(mask[b, length:].any()):
            raise ValueError("Full attention requires nonempty, contiguous right padding")
        # Match the old extractor's float().mean(heads) reduction, one sample at a time.
        value = attention[b, :, :length, :length].float().mean(dim=0)
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError("Invalid attention probabilities")
        if not torch.allclose(value.sum(-1), torch.ones_like(value[:, 0]), atol=0.02, rtol=0.02):
            raise ValueError("Valid attention rows do not sum to approximately one")
        if bool((value.triu(1).abs() > 1e-5).any()):
            raise ValueError("Expected causal decoder attention; found future-token probability")
        rows.append(value.cpu().contiguous())
    return rows


def pack_rows(matrices, token_ids, indices, identities, dtype):
    lengths = torch.tensor([len(ids) for ids in token_ids], dtype=torch.int64)
    token_offsets = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    matrix_offsets = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.square().cumsum(0)])
    if len(matrices) != len(token_ids) or any(tuple(a.shape) != (len(t), len(t)) for a, t in zip(matrices, token_ids)):
        raise ValueError("Matrix/token sample alignment mismatch")
    return {
        "dataset_idx": torch.tensor(indices, dtype=torch.int64),
        "sample_uid": old.uid_hex_to_tensor([identities[i]["sample_uid"] for i in indices]),
        "source_line_no": torch.tensor([identities[i]["source_line_no"] for i in indices], dtype=torch.int64),
        "query_hash": old.uid_hex_to_tensor([identities[i]["query_hash"] for i in indices]),
        "valid_lengths": lengths,
        "token_offsets": token_offsets,
        "input_ids": torch.cat(token_ids).to(torch.int64),
        "matrix_offsets": matrix_offsets,
        "teacher_full_attention_mean": torch.cat([m.reshape(-1).to(dtype) for m in matrices]),
    }


def paths(root, layer, shard):
    stem = Path(root) / "full_attention" / f"layer-{layer:03d}" / f"shard-{shard:05d}"
    return str(stem.with_suffix(".safetensors")), str(stem.with_suffix(".json"))


def valid_shard(root, layer, shard, start, end, identities, manifest):
    tensor_path, sidecar_path = paths(root, layer, shard)
    try:
        sidecar = old.read_json(sidecar_path)
        if (sidecar.get("bank_fingerprint") != manifest["bank_fingerprint"]
                or sidecar.get("layer") != layer or sidecar.get("start_idx") != start
                or sidecar.get("end_idx") != end or sidecar.get("format") != FORMAT
                or sidecar.get("sha256") != old.sha256_file(tensor_path)):
            return False
        values = load_file(tensor_path)
        expected_keys = {"dataset_idx", "sample_uid", "source_line_no", "query_hash", "valid_lengths",
                         "token_offsets", "input_ids", "matrix_offsets", "teacher_full_attention_mean"}
        if set(values) != expected_keys:
            return False
        if not torch.equal(values["dataset_idx"], torch.arange(start, end)):
            return False
        if not torch.equal(values["sample_uid"], old.uid_hex_to_tensor([i["sample_uid"] for i in identities[start:end]])):
            return False
        lengths = values["valid_lengths"]
        if lengths.shape != (end-start,) or lengths.dtype != torch.int64 or bool((lengths <= 0).any()):
            return False
        if bool((lengths > manifest["max_length"]).any()):
            return False
        for name, sizes in (("token_offsets", lengths), ("matrix_offsets", lengths.square())):
            expected = torch.cat([torch.zeros(1, dtype=torch.int64), sizes.cumsum(0)])
            if values[name].dtype != torch.int64 or not torch.equal(values[name], expected):
                return False
        tokens, matrices = values["input_ids"], values["teacher_full_attention_mean"]
        if tokens.dtype != torch.int64 or tokens.shape != (int(lengths.sum()),):
            return False
        if matrices.dtype != getattr(torch, manifest["storage_dtype"]) or matrices.shape != (int(lengths.square().sum()),):
            return False
        return bool(torch.isfinite(matrices).all()) and not bool((matrices < 0).any())
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def save_shard(root, layer, shard, start, end, payload, manifest):
    tensor_path, sidecar_path = paths(root, layer, shard)
    old.atomic_safetensors(tensor_path, payload, {"format": FORMAT, "bank_fingerprint": manifest["bank_fingerprint"]})
    old.atomic_json(sidecar_path, {
        "format": FORMAT, "bank_fingerprint": manifest["bank_fingerprint"], "layer": layer,
        "start_idx": start, "end_idx": end, "sha256": old.sha256_file(tensor_path),
    })


def run(args):
    if args.batch_size < 1 or args.shard_size < 1 or (args.max_samples is not None and args.max_samples < 1):
        raise ValueError("batch_size, shard_size and max_samples must be positive")
    rank, world, local_rank = old.distributed_context()
    device = torch.device("cuda", local_rank)
    old.seed_everything(args.seed + rank)
    checkpoint = os.path.abspath(args.checkpoint_dir or old.DEFAULT_TEACHER_CHECKPOINTS[args.dataset])
    config = old.find_teacher_config(checkpoint)
    if config.get("dataset") != args.dataset:
        raise ValueError("Checkpoint dataset mismatch")
    fingerprint = old.teacher_checkpoint_fingerprint(checkpoint)
    config["model_path"] = fingerprint["base_model_path"]
    data_cfg = old.DATASET_ZOO[args.dataset]
    train_path = os.path.join(args.data_dir or data_cfg["path"], data_cfg["train_file"])
    meta = old.load_preprocess_meta(train_path)
    levels = meta["num_levels"]
    offsets = old.build_level_offsets(meta["level_classes"])
    mapping = old.normalize_layer_label2id(meta["meta"])
    if (int(config["num_levels"]) != levels or config["level_classes"] != meta["level_classes"]
            or config["level_offsets"] != offsets
            or mapping != old.normalize_layer_label2id({"layer_label2id": config["layer_label2id"]})):
        raise ValueError("Teacher and dataset label metadata mismatch")
    max_length = int(config.get("max_length", data_cfg["max_length"]))
    truncate_mode = str(config.get("truncate_mode", "head"))
    tokenizer_path = checkpoint if os.path.exists(os.path.join(checkpoint, "tokenizer_config.json")) else config["model_path"]
    tokenizer = old.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if tokenizer_path != checkpoint:
        old.add_level_special_tokens(tokenizer, old.LEVEL_TOKENS[:levels])
    level_ids = tokenizer.convert_tokens_to_ids(old.LEVEL_TOKENS[:levels])
    if config.get("level_token_ids") and config["level_token_ids"] != level_ids:
        raise ValueError("Level token IDs mismatch")
    dataset = old.HierarchicalDataset(train_path, tokenizer, levels, meta["total_classes"], offsets,
                                      max_length, truncate_mode, strict=True)
    identities = old.build_sample_identities(train_path, args.dataset, "train", levels)
    if len(dataset) != len(identities) or not len(dataset):
        raise ValueError("Empty dataset or dataset/sample identity mismatch")
    count = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    model = old.load_teacher(checkpoint, config, tokenizer, True, device)
    model.eval()
    layer_count = len(old.find_decoder_layers(model.backbone))
    layers = resolve_layers(args.layers, layer_count)
    root = os.path.abspath(args.output_dir or os.path.join(checkpoint, "teacher_knowledge_full_attention"))
    manifest = {
        "format": FORMAT, "schema_version": 1, "dataset": args.dataset, "split": "train",
        "checkpoint_dir": checkpoint, "teacher_checkpoint_fingerprint": fingerprint["sha256"],
        "data_sha256": old.sha256_file(train_path), "tokenizer_sha256": old.tokenizer_fingerprint(tokenizer),
        "num_samples": count, "source_num_samples": len(dataset), "max_length": max_length,
        "truncate_mode": truncate_mode, "num_levels": levels, "level_classes": meta["level_classes"],
        "level_offsets": offsets, "teacher_level_token_ids": level_ids,
        "layer_ids": layers, "num_decoder_layers": layer_count, "knowledge_types": ["full_attention"],
        "attention_distribution": "post_softmax_no_additional_normalization",
        "attention_scope": "all_valid_tokens_including_level_tokens_to_all_valid_tokens",
        "attention_head_reduction": "float32_mean", "causal_mask_preserved": True,
        "storage_dtype": args.storage_dtype, "packing": "flat_row_major_squares_with_offsets",
        "batch_size": args.batch_size, "shard_size": args.shard_size,
        "num_shards": (count + args.shard_size - 1)//args.shard_size, "seed": args.seed, "world_size": world,
        "model_dtype": str(model.backbone.dtype), "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "peft_version": importlib.metadata.version("peft"),
        "reference_script_sha256": old.sha256_file(old.__file__), "extractor_script_sha256": old.sha256_file(__file__),
    }
    manifest["bank_fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = os.path.join(root, "manifest.json")
    error = None
    if rank == 0:
        try:
            if os.path.exists(manifest_path):
                previous = old.read_json(manifest_path)
                if previous.get("bank_fingerprint") != manifest["bank_fingerprint"]:
                    raise ValueError("Incompatible existing output; use a new full-attention directory")
            elif os.path.isdir(root) and any(Path(root).iterdir()):
                raise ValueError("Nonempty output directory without manifest")
            old.atomic_json(manifest_path, {**manifest, "status": "extracting"})
        except Exception as exc:
            error = str(exc)
    synchronized_error(error, device)
    if rank == 0:
        upper_bytes = count * len(layers) * max_length**2 * (4 if args.storage_dtype == "float32" else 2)
        print(f"Full attention matrix storage upper bound (before removing padding): {upper_bytes/2**30:.2f} GiB")

    validations = {}
    sample = dataset[0]
    for layer in layers:
        capture = old.LayerAttentionCapture(model.backbone, layer)
        error = None
        try:
            validations[str(layer)] = old.validate_attention_replay(
                model.backbone, capture, sample["input_ids"][None].to(device),
                sample["attention_mask"][None].to(device), levels)
        except Exception as exc:
            error = f"Layer {layer} replay validation failed: {exc}"
        finally:
            capture.clear()
            capture.close()
        synchronized_error(error, device)
    manifest["attention_replay_validation"] = validations
    context, captures, handles = {}, [], []

    def make_hook(layer, capture):
        def hook(module, inputs, output):
            if capture._is_replaying:
                return
            try:
                if layer in context["pending"]:
                    rows = full_attention_rows(capture.replay(), context["mask"])
                    context["matrices"][layer].extend([r.to(getattr(torch, args.storage_dtype)) for r in rows])
            finally:
                capture.clear()
        return hook

    error = None
    try:
        for layer in layers:
            capture = old.LayerAttentionCapture(model.backbone, layer)
            captures.append(capture)
            handles.append(capture.self_attn.register_forward_hook(make_hook(layer, capture)))
        for shard in tqdm(range(rank, manifest["num_shards"], world), disable=rank != 0, desc="Full attention shards"):
            start, end = shard * args.shard_size, min(count, (shard+1)*args.shard_size)
            pending = [layer for layer in layers if args.overwrite_shards or not valid_shard(root, layer, shard, start, end, identities, manifest)]
            if not pending:
                continue
            context.update(pending=pending, matrices={layer: [] for layer in pending})
            token_ids, indices = [], []
            loader = DataLoader(Subset(dataset, range(start, end)), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, collate_fn=old.trim_right_padding_collator)
            for batch in loader:
                ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
                context["mask"] = mask
                indices.extend(batch["dataset_idx"].tolist())
                for b, length in enumerate(mask.sum(-1).tolist()):
                    valid_ids = ids[b, :int(length)].cpu()
                    if valid_ids[-levels:].tolist() != level_ids:
                        raise ValueError("Level tokens must be the final valid tokens")
                    token_ids.append(valid_ids)
                with torch.inference_mode():
                    old.extract_model_outputs(model, ids, mask)
            if indices != list(range(start, end)):
                raise ValueError("Shard sample ordering mismatch")
            for layer in pending:
                payload = pack_rows(context["matrices"][layer], token_ids, indices, identities, getattr(torch, args.storage_dtype))
                save_shard(root, layer, shard, start, end, payload, manifest)
                context["matrices"][layer].clear()
                del payload
            context.clear()
    except Exception as exc:
        error = f"Full attention extraction failed: {exc}"
    finally:
        for handle in handles:
            handle.remove()
        for capture in captures:
            capture.clear()
            capture.close()
    synchronized_error(error, device)
    error = None
    if rank == 0:
        try:
            missing = []
            for shard in range(manifest["num_shards"]):
                start, end = shard*args.shard_size, min(count, (shard+1)*args.shard_size)
                missing.extend([f"layer-{layer:03d}/shard-{shard:05d}" for layer in layers
                                if not valid_shard(root, layer, shard, start, end, identities, manifest)])
            old.atomic_json(manifest_path, {**manifest, "status": "incomplete" if missing else "complete",
                                          "invalid_or_missing_shards": missing})
            if missing:
                raise RuntimeError(f"Invalid/missing shards: {len(missing)}")
            print(f"Full attention extraction complete: {root}")
        except Exception as exc:
            error = str(exc)
    synchronized_error(error, device)


if __name__ == "__main__":
    try:
        run(parse_args())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
