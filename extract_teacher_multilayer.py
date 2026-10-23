"""Per-layer teacher bank; `--layers original` delegates to the frozen user script.

See MULTILAYER_EXTRACTION.md for semantics, examples and compatibility checks.

CUDA_VISIBLE_DEVICES=0 \
python extract_teacher_multilayer.py \
  --dataset wos \
  --checkpoint_dir /data1/lwq/agt_plus/output_plus/wos_qwen8b_local0_global1_collator-epoch7/best_global_macro \
  --output_dir /data1/lwq/agt_plus/output_plus/wos_qwen8b_local0_global1_collator-epoch7/best_global_macro/teacher_knowledge_all \
  --layers all \
  --knowledge_types attention
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

try:
    import extract_teacher_reference as old
except ModuleNotFoundError as exc:
    if exc.name != "extract_teacher_reference":
        raise
    import extract_teacher as old

FORMAT = "per_layer_teacher_bank_v1"


def merge_manifest(existing, current):
    """Keep legacy shard identity; allow supplemental types, not changed semantics.

    Code hashes are provenance. FORMAT/schema and all extraction settings remain
    the compatibility contract, so upgrading this orchestration fix does not
    invalidate already extracted v1 tensors.
    """
    if existing:
        mutable = {"knowledge_types", "extractor_script_sha256", "bank_fingerprint"}
        mismatches = [key for key in current if key not in mutable
                      and existing.get(key) != current[key]]
        if mismatches:
            detail = "\n".join(f"{key}: saved={existing.get(key)!r}, requested={current[key]!r}"
                               for key in mismatches)
            raise RuntimeError("Output directory has incompatible extraction settings:\n" + detail)
        if not existing.get("bank_fingerprint"):
            raise RuntimeError("Existing manifest has no bank_fingerprint")
    result = {**(existing or current)}
    result["knowledge_types"] = sorted(set(result.get("knowledge_types", [])) | set(current["knowledge_types"]))
    history = list(result.get("extraction_requests", []))
    request = {"knowledge_types": current["knowledge_types"],
               "extractor_script_sha256": current["extractor_script_sha256"],
               "attention_backend_request": "eager" if "attention" in current["knowledge_types"] else "model_default"}
    if request not in history:
        history.append(request)
    result["extraction_requests"] = history
    result["status"] = "extracting"
    return result


def bank_modules(types, layers):
    modules = ["sample_index"]
    if "logits" in types:
        modules.append("logits")
    for kind in ("features", "attention"):
        if kind in types:
            modules.extend(f"{kind}/layer-{layer:03d}" for layer in layers)
    return modules


def resolve_layers(spec: str, count: int) -> list[int]:
    if spec == "all":
        return list(range(count))
    if spec.startswith("last:"):
        k = int(spec.split(":", 1)[1])
        if not 1 <= k <= count:
            raise ValueError(f"last:K requires 1 <= K <= {count}")
        return list(range(count - k, count))
    values = [int(s.strip()) for s in spec.split(",") if s.strip()]
    resolved = [i if i >= 0 else count + i for i in values]
    if not resolved or any(i < 0 or i >= count for i in resolved):
        raise ValueError(f"Invalid decoder layers {spec!r}; model has {count} layers")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate decoder layers are not allowed")
    return sorted(resolved)


def parse_options():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--layers", default="original",
                        help="original | all | last:4 | 0,8,16,24 (zero-based decoder indices)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Multilayer-only prefix for smoke tests, in a separate output directory")
    options, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        print("Multilayer options: --layers original|all|last:K|i,j,k; --max_samples N\n")
    saved = sys.argv
    try:
        sys.argv = [saved[0], *remaining]
        args = old.parse_args()
    finally:
        sys.argv = saved
    return options, args, remaining


class FeatureCapture:
    """Copy only level-token vectors to CPU; never retain whole-layer GPU outputs.

    Intermediate layers: decoder output before backbone final norm.
    Last layer: backbone last_hidden_state, already final-normalized.
    Apply teacher post_ln to each, exactly as the reference's feature definition.
    """
    def __init__(self, model, layer_ids):
        self.model = model
        self.values = {}
        self.input_ids = None
        self.handles = []
        decoder = old.find_decoder_layers(model.backbone)
        self.last = len(decoder) - 1
        for index in layer_ids:
            if index != self.last:
                self.handles.append(decoder[index].register_forward_hook(self._hook(index)))
        if self.last in layer_ids:
            self.handles.append(model.backbone.register_forward_hook(self._final_hook))

    def _store(self, index, hidden):
        features = self.model._extract_level_features(hidden, self.input_ids)
        value = torch.stack([self.model.post_ln(f) for f in features], dim=1)
        self.values[index] = value.to(torch.float16).cpu()

    def _hook(self, index):
        def capture(_module, _args, output):
            self._store(index, output[0] if isinstance(output, (tuple, list)) else output)
        return capture

    def _final_hook(self, _module, _args, output):
        self._store(self.last, output.last_hidden_state)

    def begin(self, input_ids):
        self.values.clear()
        self.input_ids = input_ids

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.values.clear()
        self.input_ids = None


def dense_attention(attention, input_ids, mask, level_ids, offsets, max_query_length):
    """Store raw head-mean text weights in FP32, before text renormalization.

    This preserves mass for average-then-normalize; padding slots are zero.
    The head reduction deliberately matches the reference arithmetic.
    """
    mean = attention.float().mean(dim=1)
    batch = input_ids.shape[0]
    weights = torch.zeros(batch, len(level_ids), max_query_length, dtype=torch.float32)
    token_ids = torch.full((batch, max_query_length), -1, dtype=torch.int32)
    char_offsets = torch.full((batch, max_query_length, 2), -1, dtype=torch.int32)
    lengths = mask.sum(dim=1).cpu().to(torch.int32) - len(level_ids)
    for b, length in enumerate(lengths.tolist()):
        if not 0 < length <= max_query_length:
            raise RuntimeError("Invalid query length")
        valid = length + len(level_ids)
        if mask[b, :valid].sum().item() != valid or mask[b, valid:].sum().item() != 0:
            raise RuntimeError("Dense attention requires contiguous right padding")
        if input_ids[b, length:valid].tolist() != list(level_ids):
            raise RuntimeError("Level tokens must be the final valid tokens")
        for l, token_id in enumerate(level_ids):
            matches = (input_ids[b, :valid] == token_id).nonzero(as_tuple=True)[0]
            if len(matches) != 1:
                raise RuntimeError("Level tokens must occur exactly once")
            weights[b, l, :length] = mean[b, int(matches[0]), :length].clamp_min(0).cpu()
        token_ids[b, :length] = input_ids[b, :length].to(torch.int32).cpu()
        if offsets is not None:
            char_offsets[b, :length] = torch.tensor(offsets[b], dtype=torch.int32)
    return {"teacher_text_attention_raw_mean": weights, "query_lengths": lengths,
            "query_token_ids": token_ids, "query_char_offsets": char_offsets,
            "offsets_available": torch.full((batch,), offsets is not None, dtype=torch.bool)}


def bank_schema(module, count, manifest):
    kind = module.split("/")[0]
    schema = old.expected_shard_schema(kind, count, manifest)
    if kind == "attention":
        length = manifest["max_length"] - manifest["num_levels"]
        schema.update({
            "teacher_text_attention_raw_mean": (torch.float32, (count, manifest["num_levels"], length)),
            "query_lengths": (torch.int32, (count,)),
            "query_token_ids": (torch.int32, (count, length)),
            "query_char_offsets": (torch.int32, (count, length, 2)),
            "offsets_available": (torch.bool, (count,)),
        })
    return schema


def valid_shard(root, module, shard, start, end, uids, manifest):
    path, sidecar = old.shard_paths(root, module, shard)
    try:
        meta = old.read_json(sidecar)
        if (meta["bank_fingerprint"] != manifest["bank_fingerprint"]
                or meta["module"] != module or meta["start_idx"] != start
                or meta["end_idx"] != end or meta["sha256"] != old.sha256_file(path)):
            return False
        with safe_open(path, framework="pt", device="cpu") as handle:
            schema = bank_schema(module, end - start, manifest)
            if set(handle.keys()) != set(schema):
                return False
            for key, (dtype, shape) in schema.items():
                tensor = handle.get_tensor(key)
                if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                    return False
                old.assert_finite_tensor(key, tensor)
            return (torch.equal(handle.get_tensor("sample_uid"), uids)
                    and torch.equal(handle.get_tensor("dataset_idx"), torch.arange(start, end)))
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def save_shard(root, module, shard, start, end, payload, manifest):
    path, sidecar = old.shard_paths(root, module, shard)
    schema = bank_schema(module, end - start, manifest)
    if set(payload) != set(schema):
        raise RuntimeError(f"Unexpected keys in {module}")
    for key, (dtype, shape) in schema.items():
        if payload[key].dtype != dtype or tuple(payload[key].shape) != shape:
            raise RuntimeError(f"Invalid shape/dtype for {module}/{key}")
        old.assert_finite_tensor(f"{module}/{key}", payload[key])
    metadata = {"format": FORMAT, "bank_fingerprint": manifest["bank_fingerprint"], "module": module}
    old.atomic_safetensors(path, payload, metadata)
    old.atomic_json(sidecar, {**metadata, "start_idx": start, "end_idx": end,
                              "sha256": old.sha256_file(path)})


def synchronized_error(message, device):
    """All ranks agree on validation/setup failures before further collectives."""
    if dist.is_initialized():
        messages = [None] * dist.get_world_size()
        dist.all_gather_object(messages, message)
        message = next((m for m in messages if m), None)
    if message:
        raise RuntimeError(message)


def multilayer_main(options, args):
    if args.batch_size <= 0 or args.shard_size <= 0 or args.attention_topk <= 0:
        raise ValueError("batch_size, shard_size and attention_topk must be positive")
    if options.max_samples is not None and options.max_samples <= 0:
        raise ValueError("max_samples must be positive")
    rank, world, local_rank = old.distributed_context()
    device = torch.device("cuda", local_rank)
    old.seed_everything(args.seed + rank)
    types = old.parse_knowledge_types(args.knowledge_types)
    checkpoint = os.path.abspath(args.checkpoint_dir or old.DEFAULT_TEACHER_CHECKPOINTS[args.dataset])
    config = old.find_teacher_config(checkpoint)
    if config.get("dataset") != args.dataset:
        raise RuntimeError("Checkpoint dataset mismatch")
    # Fingerprint exactly the same source weights as the reference script.
    shared = [None, None, None]
    if rank == 0:
        try:
            shared[0] = old.teacher_checkpoint_fingerprint(checkpoint)
            shared[1] = old.read_head_training_status(checkpoint, config)
        except Exception as exc:
            shared[2] = str(exc)
    if dist.is_initialized():
        dist.broadcast_object_list(shared, src=0)
    fingerprint, status, error = shared
    if error:
        raise RuntimeError(error)
    if "logits" in types and not status["has_trained_global_head"]:
        raise RuntimeError("Logit extraction requires a trained global head")
    config["model_path"] = fingerprint["base_model_path"]
    dataset_config = old.DATASET_ZOO[args.dataset]
    train_path = os.path.join(args.data_dir or dataset_config["path"], dataset_config["train_file"])
    meta = old.load_preprocess_meta(train_path)
    levels = meta["num_levels"]
    if int(config["num_levels"]) != levels:
        raise RuntimeError("Teacher and data num_levels mismatch")
    mapping = old.normalize_layer_label2id(meta["meta"])
    if mapping != old.normalize_layer_label2id({"layer_label2id": config["layer_label2id"]}):
        raise RuntimeError("Teacher label mapping differs from data")
    offsets = old.build_level_offsets(meta["level_classes"])
    if offsets != config["level_offsets"] or meta["level_classes"] != config["level_classes"]:
        raise RuntimeError("Teacher class counts/offsets differ from data")
    max_length = int(config.get("max_length", dataset_config["max_length"]))
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
        raise RuntimeError("Level token IDs mismatch")
    dataset = old.HierarchicalDataset(train_path, tokenizer, levels, meta["total_classes"],
                                      offsets, max_length, truncate_mode, strict=True)
    identities = old.build_sample_identities(train_path, args.dataset, "train", levels)
    if len(dataset) != len(identities):
        raise RuntimeError("Dataset and identity count mismatch")
    count = len(dataset) if options.max_samples is None else min(options.max_samples, len(dataset))
    model = old.load_teacher(checkpoint, config, tokenizer, "attention" in types, device)
    if any(bool(getattr(model, k)) != bool(v) for k, v in status.items()):
        raise RuntimeError("Loaded head training status mismatch")
    decoder_count = len(old.find_decoder_layers(model.backbone))
    layer_ids = resolve_layers(options.layers, decoder_count)
    root = os.path.abspath(args.output_dir or os.path.join(checkpoint, "teacher_knowledge_multilayer_train"))
    manifest_path = os.path.join(root, "manifest.json")
    manifest = {
        "format": FORMAT, "schema_version": 1, "dataset": args.dataset, "split": "train",
        "data_sha256": old.sha256_file(train_path), "num_samples": count, "source_num_samples": len(dataset),
        "shard_size": args.shard_size, "num_shards": (count + args.shard_size - 1) // args.shard_size,
        "teacher_checkpoint_fingerprint": fingerprint["sha256"],
        "reference_script_sha256": old.sha256_file(old.__file__),
        "extractor_script_sha256": old.sha256_file(__file__),
        "tokenizer_sha256": old.tokenizer_fingerprint(tokenizer),
        "num_levels": levels, "level_classes": meta["level_classes"], "level_offsets": offsets,
        "total_classes": meta["total_classes"], "layer_label2id": mapping,
        "teacher_level_token_ids": level_ids, "teacher_hidden_size": int(model.backbone.config.hidden_size),
        "max_length": max_length, "truncate_mode": truncate_mode, "layer_ids": layer_ids,
        "num_decoder_layers": decoder_count, "reference_feature_layer": old.configured_feature_layer(config),
        "knowledge_types": sorted(types), "attention_topk": args.attention_topk,
        "feature_type": "native_hidden_state_then_teacher_post_ln",
        "last_layer_final_norm": True, "intermediate_layer_final_norm": False,
        "attention_distribution": "raw_head_mean_text_fp32_and_reference_topk_fp16",
        "seed": args.seed, "batch_size": args.batch_size, "world_size": world,
        "model_dtype": str(model.backbone.dtype), "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "peft_version": importlib.metadata.version("peft"),
        **status,
    }
    manifest["bank_fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    error = None
    if rank == 0:
        try:
            if os.path.exists(manifest_path):
                existing = old.read_json(manifest_path)
                manifest = merge_manifest(existing, manifest)
            elif os.path.isdir(root) and any(Path(root).iterdir()):
                raise RuntimeError("Refusing a non-empty output directory without a bank manifest")
            else:
                manifest = merge_manifest({}, manifest)
            old.atomic_json(manifest_path, {**manifest, "status": "extracting"})
        except Exception as exc:
            error = str(exc)
    synchronized_error(error, device)
    # Every rank must use the same retained legacy fingerprint and union of types.
    manifest = old.read_json(manifest_path)
    # Validate each selected attention layer using the unchanged reference check.
    # Capture/replay one layer at a time during the real forward: no all-layer
    # attention collection and no simultaneous copies of full hidden states.
    if "attention" in types:
        sample = dataset[0]
        validations = {}
        for layer in layer_ids:
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
    modules = bank_modules(types, layer_ids)
    all_modules = bank_modules(manifest["knowledge_types"], layer_ids)
    feature_capture = FeatureCapture(model, layer_ids) if "features" in types else None
    captures, handles = [], []
    attention_values = {}
    context = {}

    def attention_hook(layer, capture):
        def hook(_module, _inputs, _output):
            if capture._is_replaying:
                return
            value = capture.replay()
            args_topk = (value, context["ids"], context["mask"], level_ids,
                         args.attention_topk, context["offsets"])
            payload = {k: v.cpu() for k, v in old.topk_attention_knowledge(*args_topk).items()}
            payload.update(dense_attention(value, context["ids"], context["mask"], level_ids,
                                           context["offsets"], max_length - levels))
            attention_values[layer] = payload
            capture.clear()
        return hook

    try:
        if "attention" in types:
            for layer in layer_ids:
                capture = old.LayerAttentionCapture(model.backbone, layer)
                captures.append(capture)
                handles.append(capture.self_attn.register_forward_hook(attention_hook(layer, capture)))
        for shard in tqdm(range(rank, manifest["num_shards"], world), disable=rank != 0, desc="Multilayer shards"):
            start, end = shard * args.shard_size, min(count, (shard + 1) * args.shard_size)
            uids = old.uid_hex_to_tensor([i["sample_uid"] for i in identities[start:end]])
            pending = [m for m in modules if args.overwrite_shards
                       or not valid_shard(root, m, shard, start, end, uids, manifest)]
            if not pending:
                continue
            accum = {m: {} for m in pending}

            def append(module, payload):
                if module not in accum:
                    return
                for key, value in payload.items():
                    accum[module].setdefault(key, []).append(value.cpu())

            loader = DataLoader(Subset(dataset, range(start, end)), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True,
                                collate_fn=old.trim_right_padding_collator)
            for batch in loader:
                indices = batch["dataset_idx"].tolist()
                rows = [identities[i] for i in indices]
                ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
                context.update(ids=ids, mask=mask)
                if "attention" in types:
                    context["offsets"] = old.encode_offsets(tokenizer, [dataset.data[i]["query"] for i in indices],
                                                            ids, mask, levels, truncate_mode)
                attention_values.clear()
                if feature_capture:
                    feature_capture.begin(ids)
                with torch.inference_mode():
                    outputs, reference_features = old.extract_model_outputs(model, ids, mask)
                common = {"dataset_idx": torch.tensor(indices, dtype=torch.int64),
                          "sample_uid": old.uid_hex_to_tensor([r["sample_uid"] for r in rows])}
                append("sample_index", {**common, "source_line_no": torch.tensor([r["source_line_no"] for r in rows], dtype=torch.int64),
                                        "query_hash": old.uid_hex_to_tensor([r["query_hash"] for r in rows])})
                if "logits" in types:
                    logits = {"teacher_global_logits": outputs["global_logits"].to(torch.float16).cpu()}
                    if model.has_trained_local_heads:
                        logits["teacher_local_logits"] = torch.cat(outputs["logits_list"], dim=-1).to(torch.float16).cpu()
                    append("logits", {**common, **logits})
                if feature_capture:
                    configured = old.configured_feature_layer(config)
                    original_layer = configured if configured >= 0 else decoder_count + configured
                    # For the original layer, use the actual legacy returned tensor.
                    if original_layer in layer_ids:
                        feature_capture.values[original_layer] = reference_features.to(torch.float16).cpu()
                    if set(feature_capture.values) != set(layer_ids):
                        raise RuntimeError("Not every requested feature layer was captured")
                    for layer in layer_ids:
                        append(f"features/layer-{layer:03d}", {**common, "teacher_level_features": feature_capture.values[layer]})
                    feature_capture.values.clear()
                    feature_capture.input_ids = None
                if "attention" in types:
                    if set(attention_values) != set(layer_ids):
                        raise RuntimeError("Not every requested attention layer was captured")
                    for layer in layer_ids:
                        append(f"attention/layer-{layer:03d}", {**common, **attention_values[layer]})
                del outputs, reference_features
                context.clear()
                attention_values.clear()
            for module in pending:
                payload = {k: torch.cat(v, dim=0) for k, v in accum[module].items()}
                if not torch.equal(payload["dataset_idx"], torch.arange(start, end)) or not torch.equal(payload["sample_uid"], uids):
                    raise RuntimeError("Shard sample ordering mismatch")
                save_shard(root, module, shard, start, end, payload, manifest)
                accum[module].clear()
        old.barrier()
        error = None
        if rank == 0:
            try:
                missing = []
                for shard in range(manifest["num_shards"]):
                    start, end = shard * args.shard_size, min(count, (shard + 1) * args.shard_size)
                    uids = old.uid_hex_to_tensor([i["sample_uid"] for i in identities[start:end]])
                    missing.extend(f"{m}/shard-{shard:05d}" for m in all_modules
                                   if not valid_shard(root, m, shard, start, end, uids, manifest))
                old.atomic_json(manifest_path, {**manifest, "status": "incomplete" if missing else "complete",
                                                "invalid_or_missing_shards": missing})
                if missing:
                    error = f"Invalid/missing shards: {len(missing)}"
                else:
                    print(f"Multilayer extraction complete: {root}")
            except Exception as exc:
                error = str(exc)
        synchronized_error(error, device)
    finally:
        for handle in handles:
            handle.remove()
        for capture in captures:
            capture.clear()
            capture.close()
        if feature_capture:
            feature_capture.close()


def main():
    options, args, remaining = parse_options()
    if options.layers == "original":
        if options.max_samples is not None:
            raise ValueError("--max_samples is only for per-layer mode; compatibility mode runs the unchanged reference")
        if args.output_dir is None:
            checkpoint = args.checkpoint_dir or old.DEFAULT_TEACHER_CHECKPOINTS[args.dataset]
            remaining.extend(["--output_dir", os.path.join(checkpoint, "teacher_knowledge_compat_train")])
        sys.argv = [sys.argv[0], *remaining]
        old.main()
    else:
        if args.allow_full_attention_fallback:
            raise ValueError("Multilayer mode requires validated replay; full-attention fallback is only supported with --layers original")
        try:
            multilayer_main(options, args)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
