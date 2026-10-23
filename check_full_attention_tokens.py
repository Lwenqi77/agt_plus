"""Read-only audit of token IDs saved with full attention (no model/GPU needed).

Reads the last saved layer by default, all its sample shards. Does not load the
large attention tensor or modify any knowledge files. Uses the extraction tokenizer.
检查是full-att都包含什么token
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def tokenizer_fingerprint(tokenizer):
    # Same definition as extract_teacher.py; no model/PEFT imports needed.
    backend = getattr(tokenizer, "backend_tokenizer", None)
    payload = {"class": type(tokenizer).__name__,
               "vocab": sorted((str(t), int(i)) for t, i in tokenizer.get_vocab().items()),
               "backend_tokenizer": backend.to_str() if backend is not None and hasattr(backend, "to_str") else None,
               "special_tokens_map": tokenizer.special_tokens_map}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def token_catalog(tokenizer):
    special = set(int(i) for i in tokenizer.all_special_ids)
    for index, token in getattr(tokenizer, "added_tokens_decoder", {}).items():
        if getattr(token, "special", False):
            special.add(int(index))
    vocab = {int(index): str(token) for token, index in tokenizer.get_vocab().items()}
    # Also surface Qwen control-marker tokens even if not registered in the
    # current special_tokens_map (some tokenizer versions expose a smaller list).
    markers = {i for i, token in vocab.items() if token.startswith("<|") and token.endswith("|>")}
    return special, markers, vocab


def inspect_tokens(tokens, level_ids, special_ids, marker_ids, vocab_ids):
    if not level_ids or len(set(level_ids)) != len(level_ids):
        raise ValueError("Manifest must contain distinct level token IDs")
    count = len(level_ids)
    text = tokens[:-count] if len(tokens) >= count else []
    level_set = set(level_ids)
    return {
        "tail_ok": len(tokens) > count and tokens[-count:] == list(level_ids),
        "extra_special": Counter(i for i in tokens if i in special_ids and i not in level_set),
        "extra_markers": Counter(i for i in tokens if i in marker_ids and i not in level_set),
        "levels_in_text": Counter(i for i in text if i in level_set),
        "unknown_ids": Counter(i for i in tokens if i not in vocab_ids),
    }


def audit_bank(root, manifest, layer, tokenizer):
    from safetensors import safe_open
    special, markers, vocab = token_catalog(tokenizer)
    levels = list(manifest["teacher_level_token_ids"])
    counts = {key: Counter() for key in ("extra_special", "extra_markers", "levels_in_text", "unknown_ids")}
    samples = Counter()
    examples = []
    sample_count = int(manifest["num_samples"])
    shard_size = int(manifest["shard_size"])
    if shard_size <= 0 or sample_count <= 0 or manifest["num_shards"] != (sample_count+shard_size-1)//shard_size:
        raise ValueError("Invalid manifest sample/shard counts")
    for shard in range(manifest["num_shards"]):
        path = Path(root)/"full_attention"/f"layer-{layer:03d}"/f"shard-{shard:05d}.safetensors"
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if (handle.metadata() or {}).get("bank_fingerprint") != manifest["bank_fingerprint"]:
                raise ValueError(f"Manifest/shard fingerprint mismatch: {path}")
            indices = handle.get_tensor("dataset_idx").tolist()
            lengths = handle.get_tensor("valid_lengths").tolist()
            offsets = handle.get_tensor("token_offsets").tolist()
            matrix_offsets = handle.get_tensor("matrix_offsets").tolist()
            ids = handle.get_tensor("input_ids").tolist()
            matrix_shape = handle.get_slice("teacher_full_attention_mean").get_shape()
        start, end = shard*shard_size, min(sample_count,(shard+1)*shard_size)
        if indices != list(range(start,end)) or len(lengths) != len(indices):
            raise ValueError(f"Sample index/count mismatch: {path}")
        if (len(offsets) != len(indices)+1 or offsets[0] != 0 or offsets[-1] != len(ids)
                or len(matrix_offsets) != len(indices)+1 or matrix_offsets[0] != 0
                or matrix_shape != [matrix_offsets[-1]]):
            raise ValueError(f"Invalid token/matrix offsets: {path}")
        for row, index in enumerate(indices):
            length = lengths[row]
            if (not 0 < length <= manifest["max_length"] or offsets[row+1]-offsets[row] != length
                    or matrix_offsets[row+1]-matrix_offsets[row] != length*length):
                raise ValueError(f"Token/matrix lengths mismatch at sample {index}")
            result = inspect_tokens(ids[offsets[row]:offsets[row+1]], levels, special, markers, vocab)
            samples["checked"] += 1
            samples["invalid_level_tail"] += int(not result["tail_ok"])
            for key in counts:
                counts[key].update(result[key])
                samples[key] += int(bool(result[key]))
            if len(examples) < 10 and (not result["tail_ok"] or any(result[k] for k in counts)):
                examples.append({"dataset_idx": index, "tail_ok": result["tail_ok"],
                                 "hits": {k: dict(result[k]) for k in counts if result[k]}})
    clean = samples["invalid_level_tail"] == 0 and not any(counts.values())
    return {
        "scanned_layer": layer, "indexing": "zero_based", "checked_samples": samples["checked"],
        "num_shards": manifest["num_shards"], "tokenizer_fingerprint_matches": True,
        "invalid_level_tail_samples": samples["invalid_level_tail"],
        "level_tokens": [{"id": i, "token": vocab.get(i)} for i in levels],
        "findings": {key: {"samples": samples[key], "occurrences": [
            {"id": i, "token": vocab.get(i), "count": n} for i,n in sorted(value.items())]}
            for key,value in counts.items()},
        "examples": examples,
        "result": "PASS_ONLY_TEXT_AND_LEVEL_TOKENS" if clean else "REVIEW_TOKEN_FINDINGS",
        "scope": "All saved input IDs of this layer; ordinary text includes punctuation/subwords. "
                 "Attention values/checksums are not reread; no inference is run.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge_dir", required=True)
    parser.add_argument("--tokenizer_dir", help="Defaults to manifest checkpoint_dir; must contain the extraction tokenizer")
    parser.add_argument("--layer", type=int, help="Actual zero-based layer index; default last saved layer")
    args = parser.parse_args()
    root = Path(args.knowledge_dir)
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "full_token_attention_head_mean_v1" or manifest.get("status") != "complete":
        raise ValueError("Expected a completed full-attention bank")
    tokenizer_path = args.tokenizer_dir or manifest.get("checkpoint_dir")
    if not tokenizer_path or not Path(tokenizer_path).is_dir():
        raise ValueError("Set --tokenizer_dir to the local teacher tokenizer directory")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if tokenizer_fingerprint(tokenizer) != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer fingerprint differs from extraction; use the exact saved teacher tokenizer and environment")
    layer = max(manifest["layer_ids"]) if args.layer is None else args.layer
    if layer not in manifest["layer_ids"]:
        raise ValueError(f"Layer {layer} was not extracted")
    print(json.dumps(audit_bank(root,manifest,layer,tokenizer),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
