"""Extract generic text-mean features for the feature-carrier ablation.

One vector per sample: valid text hidden states -> FP32 token mean -> native
dtype -> existing teacher post_ln -> storage dtype. Level tokens remain inputs.
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

import extract_teacher as old

FORMAT = "teacher_text_mean_features_v1"
FEATURE_TYPE = "fp32_text_mean_then_native_dtype_then_post_ln"


def text_mean_post_ln(hidden, input_ids, attention_mask, level_ids, post_ln):
    """Pool text tokens only; preserve autograd when called by the student."""
    if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape or input_ids.shape != attention_mask.shape:
        raise ValueError("Hidden states, token IDs and mask shapes differ")
    if not level_ids or len(set(level_ids)) != len(level_ids):
        raise ValueError("Expected distinct level token IDs")
    pooled = []
    for b in range(hidden.shape[0]):
        length = int(attention_mask[b].sum().item())
        text_length = length - len(level_ids)
        if (text_length <= 0 or not bool((attention_mask[b, :length] == 1).all())
                or bool(attention_mask[b, length:].any())):
            raise ValueError("Text mean requires nonempty text and contiguous right padding")
        if input_ids[b, text_length:length].tolist() != list(level_ids):
            raise ValueError("Level tokens must be the final valid tokens in the configured order")
        for token in level_ids:
            if int((input_ids[b, :length] == token).sum()) != 1:
                raise ValueError("Each level token must occur exactly once")
        pooled.append(hidden[b, :text_length].float().mean(dim=0))
    return post_ln(torch.stack(pooled).to(hidden.dtype))


def selected_hidden(core, output):
    # Exact classification-layer selection from train_baseline.HierarchicalTeacher.
    if core.use_hidden_states:
        index = core.hidden_layer_idx if core.hidden_layer_idx < 0 else core.hidden_layer_idx + 1
        return output.hidden_states[index]
    return output.last_hidden_state


def synchronized_error(error):
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        error = next((e for e in errors if e), None)
    if error:
        raise RuntimeError(error)


def paths(root, shard):
    return old.shard_paths(str(root), "text_features", shard)


def feature_schema(count, manifest):
    return {"dataset_idx": (torch.int64, (count,)), "sample_uid": (torch.uint8, (count, 32)),
            "input_ids": (torch.int32, (count, manifest["max_length"])),
            "valid_lengths": (torch.int32, (count,)),
            "teacher_text_features": (getattr(torch, manifest["storage_dtype"]), (count, manifest["teacher_hidden_size"]))}


def valid_shard(root, shard, start, end, uids, manifest):
    path, sidecar = paths(root, shard)
    try:
        meta = old.read_json(sidecar)
        if (meta.get("bank_fingerprint") != manifest["bank_fingerprint"] or meta.get("format") != FORMAT
                or meta.get("start_idx") != start or meta.get("end_idx") != end
                or meta.get("sha256") != old.sha256_file(path)):
            return False
        values = load_file(path, device="cpu")
        schema = feature_schema(end-start, manifest)
        if set(values) != set(schema):
            return False
        for key, (dtype, shape) in schema.items():
            if values[key].dtype != dtype or tuple(values[key].shape) != shape:
                return False
        if not torch.equal(values["dataset_idx"], torch.arange(start,end)) or not torch.equal(values["sample_uid"], uids):
            return False
        if not bool(torch.isfinite(values["teacher_text_features"]).all()):
            return False
        levels = manifest["teacher_level_token_ids"]
        for b, length in enumerate(values["valid_lengths"].tolist()):
            if not len(levels) < length <= manifest["max_length"]:
                return False
            if values["input_ids"][b, length-len(levels):length].tolist() != levels:
                return False
            if bool((values["input_ids"][b, :length] < 0).any()):
                return False
            if any(int((values["input_ids"][b, :length] == token).sum()) != 1 for token in levels):
                return False
            if bool((values["input_ids"][b, length:] != -1).any()):
                return False
        return True
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def save_shard(root, shard, start, end, payload, manifest):
    path, sidecar = paths(root, shard)
    schema = feature_schema(end-start, manifest)
    if set(payload) != set(schema):
        raise ValueError("Unexpected text feature fields")
    for key, (dtype, shape) in schema.items():
        if payload[key].dtype != dtype or tuple(payload[key].shape) != shape:
            raise ValueError(f"Invalid text feature tensor: {key}")
        old.assert_finite_tensor(key, payload[key])
    metadata = {"format": FORMAT, "bank_fingerprint": manifest["bank_fingerprint"]}
    old.atomic_safetensors(path, payload, metadata)
    old.atomic_json(sidecar, {**metadata, "start_idx": start, "end_idx": end, "sha256": old.sha256_file(path)})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=sorted(old.DATASET_ZOO), default="wos")
    p.add_argument("--checkpoint_dir")
    p.add_argument("--output_dir")
    p.add_argument("--data_dir")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--shard_size", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--storage_dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--attention_backend", choices=("model_default", "eager"), default="model_default",
                   help="Match how the original teacher features were extracted")
    p.add_argument("--max_samples", type=int, help="Smoke test prefix; use a separate output directory")
    p.add_argument("--overwrite_shards", action="store_true")
    return p.parse_args()


def run(args):
    if min(args.batch_size, args.shard_size) < 1 or (args.max_samples is not None and args.max_samples < 1):
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
    levels, offsets = meta["num_levels"], old.build_level_offsets(meta["level_classes"])
    if (int(config["num_levels"]) != levels or config["level_classes"] != meta["level_classes"]
            or config["level_offsets"] != offsets
            or old.normalize_layer_label2id(meta["meta"]) != old.normalize_layer_label2id({"layer_label2id": config["layer_label2id"]})):
        raise ValueError("Teacher and dataset label metadata mismatch")
    max_length = int(config.get("max_length", data_cfg["max_length"]))
    truncate_mode = str(config.get("truncate_mode", "head"))
    tokenizer_path = checkpoint if os.path.exists(os.path.join(checkpoint,"tokenizer_config.json")) else config["model_path"]
    tokenizer = old.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if tokenizer_path != checkpoint:
        old.add_level_special_tokens(tokenizer, old.LEVEL_TOKENS[:levels])
    level_ids = tokenizer.convert_tokens_to_ids(old.LEVEL_TOKENS[:levels])
    if config.get("level_token_ids") and config["level_token_ids"] != level_ids:
        raise ValueError("Level token IDs mismatch")
    dataset = old.HierarchicalDataset(train_path,tokenizer,levels,meta["total_classes"],offsets,max_length,truncate_mode,strict=True)
    identities = old.build_sample_identities(train_path,args.dataset,"train",levels)
    if not len(dataset) or len(dataset) != len(identities):
        raise ValueError("Empty dataset or identity count mismatch")
    count = len(dataset) if args.max_samples is None else min(len(dataset),args.max_samples)
    model = old.load_teacher(checkpoint,config,tokenizer,args.attention_backend == "eager",device)
    model.eval()
    root = os.path.abspath(args.output_dir or os.path.join(checkpoint,"teacher_knowledge_text_mean"))
    manifest = {
        "format": FORMAT, "schema_version": 1, "dataset": args.dataset, "split": "train",
        "checkpoint_dir": checkpoint, "teacher_checkpoint_fingerprint": fingerprint["sha256"],
        "data_sha256": old.sha256_file(train_path), "tokenizer_sha256": old.tokenizer_fingerprint(tokenizer),
        "num_samples": count, "source_num_samples": len(dataset), "max_length": max_length, "truncate_mode": truncate_mode,
        "num_levels": levels, "level_classes": meta["level_classes"], "level_offsets": offsets,
        "teacher_level_token_ids": level_ids, "teacher_hidden_size": int(model.backbone.config.hidden_size),
        "feature_layer": old.configured_feature_layer(config), "feature_type": FEATURE_TYPE,
        "knowledge_types": ["text_features"], "storage_dtype": args.storage_dtype,
        "batch_size": args.batch_size, "shard_size": args.shard_size, "num_shards": (count+args.shard_size-1)//args.shard_size,
        "seed": args.seed, "world_size": world, "attention_backend_request": args.attention_backend,
        "model_dtype": str(model.backbone.dtype), "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"), "peft_version": importlib.metadata.version("peft"),
        "reference_script_sha256": old.sha256_file(old.__file__), "extractor_script_sha256": old.sha256_file(__file__),
    }
    manifest["bank_fingerprint"] = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    manifest_path = os.path.join(root,"manifest.json")
    error = None
    if rank == 0:
        try:
            if os.path.isfile(manifest_path):
                if old.read_json(manifest_path).get("bank_fingerprint") != manifest["bank_fingerprint"]:
                    raise ValueError("Incompatible existing text-feature bank; use a new directory")
            elif os.path.isdir(root) and any(Path(root).iterdir()):
                raise ValueError("Refusing nonempty output directory without compatible manifest")
            old.atomic_json(manifest_path,{**manifest,"status":"extracting"})
        except Exception as exc:
            error = str(exc)
    synchronized_error(error)
    context = {}

    def capture(module, inputs, output):
        context["features"] = text_mean_post_ln(selected_hidden(model,output),context["ids"],context["mask"],
                                                level_ids,model.post_ln).to(getattr(torch,args.storage_dtype)).cpu()

    handle = model.backbone.register_forward_hook(capture)
    error = None
    try:
        for shard in tqdm(range(rank,manifest["num_shards"],world),disable=rank != 0,desc="Text feature shards"):
            start, end = shard*args.shard_size, min(count,(shard+1)*args.shard_size)
            uids = old.uid_hex_to_tensor([r["sample_uid"] for r in identities[start:end]])
            if not args.overwrite_shards and valid_shard(root,shard,start,end,uids,manifest):
                continue
            chunks = {key: [] for key in feature_schema(end-start,manifest)}
            loader = DataLoader(Subset(dataset,range(start,end)),batch_size=args.batch_size,shuffle=False,
                                num_workers=args.num_workers,pin_memory=True,collate_fn=old.trim_right_padding_collator)
            for batch in loader:
                ids, mask = batch["input_ids"].to(device),batch["attention_mask"].to(device)
                indices = batch["dataset_idx"].tolist()
                context.update(ids=ids,mask=mask)
                with torch.inference_mode():
                    old.extract_model_outputs(model,ids,mask)
                token_ids = torch.full((len(indices),max_length),-1,dtype=torch.int32)
                lengths = mask.sum(-1).cpu().to(torch.int32)
                for b,length in enumerate(lengths.tolist()):
                    token_ids[b,:length] = ids[b,:length].cpu().to(torch.int32)
                payload = {"dataset_idx": torch.tensor(indices,dtype=torch.int64),
                           "sample_uid": old.uid_hex_to_tensor([identities[i]["sample_uid"] for i in indices]),
                           "input_ids": token_ids,"valid_lengths":lengths,"teacher_text_features":context["features"]}
                for key,value in payload.items():
                    chunks[key].append(value)
                context.clear()
            payload = {k:torch.cat(v) for k,v in chunks.items()}
            if not torch.equal(payload["dataset_idx"],torch.arange(start,end)) or not torch.equal(payload["sample_uid"],uids):
                raise ValueError("Text feature shard sample ordering mismatch")
            save_shard(root,shard,start,end,payload,manifest)
    except Exception as exc:
        error = str(exc)
    finally:
        handle.remove()
        context.clear()
    synchronized_error(error)
    error = None
    if rank == 0:
        try:
            missing = []
            for shard in range(manifest["num_shards"]):
                start,end = shard*args.shard_size,min(count,(shard+1)*args.shard_size)
                uids = old.uid_hex_to_tensor([r["sample_uid"] for r in identities[start:end]])
                if not valid_shard(root,shard,start,end,uids,manifest):
                    missing.append(shard)
            old.atomic_json(manifest_path,{**manifest,"status":"incomplete" if missing else "complete","invalid_or_missing_shards":missing})
            if missing:
                raise RuntimeError(f"Invalid text feature shards: {missing}")
            print(f"Text-mean feature extraction complete: {root}")
        except Exception as exc:
            error = str(exc)
    synchronized_error(error)


if __name__ == "__main__":
    try:
        run(parse_args())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
