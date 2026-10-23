"""Single-compatible and multilayer distillation. See MULTILAYER_DISTILLATION.md.

Legacy mode delegates directly to train_distill_mi_ma.main. Bank mode extends
data/feature/attention handling while inheriting evaluation and checkpoint code.
"""
from __future__ import annotations

from collections import OrderedDict
import argparse
import sys

import train_distill_mi_ma as old
from train_distill_mi_ma import *
from extract_teacher_multilayer import resolve_layers, bank_schema, valid_shard, FORMAT
from teacher_bank_ops import select_layers, mean_features, mean_attention


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--knowledge_mode", choices=["legacy", "bank"], default="legacy")
    for kind in ("feature", "attention"):
        parser.add_argument(f"--{kind}_alignment", choices=["single", "mean", "one_to_one"], default="single")
        parser.add_argument(f"--teacher_{kind}_layers", default="original",
                            help="original | last:K | random:K | comma-separated decoder indices")
        parser.add_argument(f"--student_{kind}_layers", default=None,
                            help="Decoder indices or last:K; mean supports one or multiple student layers")
    parser.add_argument("--layer_selection_seed", type=int, default=42)
    parser.add_argument("--knowledge_cache_shards", type=int, default=2,
                        help="Number of sample shards cached per DataLoader worker, across selected modules")
    extra, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        parser.print_help()
    argv = sys.argv
    try:
        sys.argv = [argv[0], *remaining]
        args = old.parse_args()
    finally:
        sys.argv = argv
    vars(args).update(vars(extra))
    args._legacy_argv = remaining
    return args


def teacher_layers(spec, manifest, seed):
    available = manifest["layer_ids"]
    if spec == "original":
        index = manifest["reference_feature_layer"]
        selected = resolve_layers(str(index), manifest["num_decoder_layers"])
    elif spec.startswith("random:"):
        selected = select_layers(available, "random", int(spec.split(":")[1]), seed=seed)
    elif spec.startswith("last:"):
        # Last K of the real teacher, not just last K of an arbitrary sparse bank.
        selected = resolve_layers(spec, manifest["num_decoder_layers"])
    else:
        selected = resolve_layers(spec, manifest["num_decoder_layers"])
    if not set(selected) <= set(available):
        raise ValueError(f"Requested teacher layers {selected} not in saved layers {available}")
    return selected


def make_plan(args, manifest):
    plan = {"teacher_checkpoint_fingerprint": manifest["teacher_checkpoint_fingerprint"],
            "bank_fingerprint": manifest["bank_fingerprint"], "selection_seed": args.layer_selection_seed,
            "feature_definition": manifest["feature_type"], "feature_projection": "shared_original_linear",
            "attention_mean": "raw_layer_mean_then_text_normalize_then_topk",
            "implementation_version": 1}
    for kind, enabled in (("feature", args.feature_kd_weight > 0),
                          ("attention", args.attention_kd_weight > 0 or args.attention_mass_kd_weight > 0)):
        if not enabled:
            plan[kind] = None
            continue
        mode = getattr(args, f"{kind}_alignment")
        layers = teacher_layers(getattr(args, f"teacher_{kind}_layers"), manifest, args.layer_selection_seed)
        if mode == "single" and len(layers) != 1:
            raise ValueError(f"{kind} single alignment requires exactly one teacher layer")
        if kind == "attention":
            validations = manifest.get("attention_replay_validation", {})
            if any(str(i) not in validations for i in layers):
                raise ValueError("Selected teacher attention layers lack replay validation records")
        plan[kind] = {"mode": mode, "teacher_layers": layers}
    return plan


def finish_student_plan(plan, args, count):
    for kind in ("feature", "attention"):
        entry = plan[kind]
        if entry is None:
            continue
        spec = getattr(args, f"student_{kind}_layers")
        if spec is None:
            if entry["mode"] == "one_to_one":
                raise ValueError(f"Specify --student_{kind}_layers for one_to_one")
            fallback = args.student_feature_layer if kind == "feature" else args.student_attention_layer
            if fallback is None:
                fallback = args.student_feature_layer
            spec = str(fallback)
        layers = resolve_layers(spec, count)
        expected = len(entry["teacher_layers"]) if entry["mode"] == "one_to_one" else 1
        student_mean = entry["mode"] == "mean"
        if not student_mean and len(layers) != expected:
            raise ValueError(f"{kind}: expected {expected} student layers, got {layers}")
        entry["student_layers"] = layers
        if student_mean and len(layers) > 1:
            entry["student_reduction"] = (
                "fp32_mean_of_post_ln_shared_projection" if kind == "feature"
                else "fp32_head_mean_then_raw_text_layer_mean_then_normalize_teacher_bins"
            )
    return plan


SPARSE_KEYS = ("topk_positions", "topk_token_ids", "topk_valid_mask", "teacher_topk_attention_mean",
               "teacher_other_attention_mean", "teacher_text_attention_mass")


def fused_attention(rows, topk):
    """One sample, layer average -> raw mass -> text normalization -> sparse bins."""
    first = rows[0]
    length = int(first["query_lengths"])
    for row in rows[1:]:
        for key in ("query_lengths", "query_token_ids", "query_char_offsets", "offsets_available"):
            if not torch.equal(row[key], first[key]):
                raise RuntimeError(f"Teacher attention layers are not aligned: {key}")
    raw = torch.stack([r["teacher_text_attention_raw_mean"] for r in rows])[:, None]
    probs, mass = mean_attention(raw, first["query_lengths"].reshape(1))
    probs, mass = probs[0], mass[0]
    levels = probs.shape[0]
    k = min(length, topk)
    values, indices = torch.topk(probs[:, :length], k, dim=-1, sorted=True)
    positions = torch.full((levels, topk), -1, dtype=torch.int32)
    token_ids = torch.full_like(positions, -1)
    valid = torch.zeros(levels, topk, dtype=torch.bool)
    weights = torch.zeros(levels, topk, dtype=torch.float32)
    positions[:, :k] = indices.int()
    token_ids[:, :k] = first["query_token_ids"][indices]
    valid[:, :k] = True
    weights[:, :k] = values
    return dict(topk_positions=positions, topk_token_ids=token_ids, topk_valid_mask=valid,
                teacher_topk_attention_mean=weights, teacher_other_attention_mean=(1 - values.sum(-1)).clamp_min(0),
                teacher_text_attention_mass=mass)


class BankKnowledge:
    def __init__(self, knowledge_dir, manifest, required_modules, expected_uids, plan, cache_shards=2):
        if cache_shards < 1:
            raise ValueError("knowledge_cache_shards must be positive")
        self.root, self.manifest, self.plan = knowledge_dir, manifest, plan
        self.cache_limit = cache_shards
        self.cache = OrderedDict()
        self.sample_count = manifest["num_samples"]
        if len(expected_uids) != self.sample_count:
            raise ValueError("Knowledge sample count differs from training data")
        if not required_modules <= set(manifest["knowledge_types"]):
            raise ValueError("Bank lacks required knowledge types")
        self.modules = ["sample_index"]
        if "logits" in required_modules:
            self.modules.append("logits")
        for kind, directory in (("feature", "features"), ("attention", "attention")):
            if plan[kind]:
                self.modules.extend(f"{directory}/layer-{i:03d}" for i in plan[kind]["teacher_layers"])
        for shard in range(manifest["num_shards"]):
            start = shard * manifest["shard_size"]
            end = min(start + manifest["shard_size"], self.sample_count)
            uids = uid_hex_to_tensor(expected_uids[start:end])
            for module in self.modules:
                if not valid_shard(self.root, module, shard, start, end, uids, manifest):
                    raise RuntimeError(f"Bank shard failed UID/schema/checksum validation: {module}/shard-{shard:05d}")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["cache"] = OrderedDict()
        return state

    def _row(self, module, idx):
        shard, offset = divmod(idx, self.manifest["shard_size"])
        if shard not in self.cache:
            # Evict before loading to bound the number of complete sample shards.
            while len(self.cache) >= self.cache_limit:
                self.cache.popitem(last=False)
            self.cache[shard] = {}
        self.cache.move_to_end(shard)
        modules = self.cache[shard]
        if module not in modules:
            path = os.path.join(self.root, module, f"shard-{shard:05d}.safetensors")
            tensors = load_file(path, device="cpu")
            if module.startswith("attention/"):
                mode = self.plan["attention"]["mode"]
                dense = ("teacher_text_attention_raw_mean", "query_lengths", "query_token_ids",
                         "query_char_offsets", "offsets_available")
                keys = dense if mode == "mean" and len(self.plan["attention"]["teacher_layers"]) > 1 else SPARSE_KEYS
            elif module.startswith("features/"):
                keys = ("teacher_level_features",)
            else:
                keys = tuple(k for k in tensors if k.startswith("teacher_"))
            modules[module] = {k: tensors[k] for k in keys}
        return {k: tensor[offset] for k, tensor in modules[module].items()}

    def item(self, idx):
        if not 0 <= idx < self.sample_count:
            raise IndexError(idx)
        result = self._row("logits", idx) if "logits" in self.modules else {}
        entry = self.plan["feature"]
        if entry:
            values = torch.stack([self._row(f"features/layer-{i:03d}", idx)["teacher_level_features"]
                                  for i in entry["teacher_layers"]])
            if entry["mode"] == "mean" and len(values) > 1:
                values = mean_features(values[:, None])[0][None]
            result["bank_features"] = values
        entry = self.plan["attention"]
        if entry:
            rows = [self._row(f"attention/layer-{i:03d}", idx) for i in entry["teacher_layers"]]
            if entry["mode"] == "mean" and len(rows) > 1:
                rows = [fused_attention(rows, self.manifest["attention_topk"])]
            for key in SPARSE_KEYS:
                result[f"bank_{key}"] = torch.stack([row[key] for row in rows])
        return result


class MultiStudent(old.DistillStudent):
    def configure_layers(self, plan):
        self.alignment_plan = plan
        self._feature_handles = []
        self._feature_values = {}
        self._feature_ids = None
        self._capture_features = False
        layers = find_decoder_layers(self.core.backbone)
        self._last_layer = len(layers) - 1
        configured = self.core.hidden_layer_idx if self.core.use_hidden_states else -1
        self._classification_layer = configured if configured >= 0 else len(layers) + configured
        wanted = plan["feature"]["student_layers"] if plan["feature"] else []
        for index in wanted:
            if index == self._classification_layer:
                continue
            if index == self._last_layer:
                self._feature_handles.append(self.core.backbone.register_forward_hook(self._final_hook))
            else:
                self._feature_handles.append(layers[index].register_forward_hook(self._layer_hook(index)))

    def _store(self, index, hidden):
        if not self._capture_features:
            return
        vectors = self.core._extract_level_features(hidden, self._feature_ids)
        features = torch.stack([self.core.post_ln(v) for v in vectors], dim=1)
        # No detach/cpu: gradients must reach the original student forward graph.
        self._feature_values[index] = self.feature_projection(features)

    def _layer_hook(self, index):
        def hook(module, args, output):
            self._store(index, output[0] if isinstance(output, (list, tuple)) else output)
        return hook

    def _final_hook(self, module, args, output):
        if self._capture_features:
            self._store(self._last_layer, output.last_hidden_state)

    def forward(self, input_ids, attention_mask, **kwargs):
        self._feature_values.clear()
        self._feature_ids = input_ids
        self._capture_features = bool(self.alignment_plan["feature"])
        try:
            outputs = super().forward(input_ids, attention_mask, **kwargs)
            if self.alignment_plan["feature"]:
                self._feature_values[self._classification_layer] = outputs["projected_level_features"]
                outputs["bank_projected_features"] = [self._feature_values[i]
                                                      for i in self.alignment_plan["feature"]["student_layers"]]
                predictions = outputs["bank_projected_features"]
                if self.alignment_plan["feature"]["mode"] == "mean" and len(predictions) > 1:
                    # Same per-layer normalization and shared projection as before.
                    # Reduce to ONE target, preserving the original feature loss.
                    outputs["bank_projected_features"] = [
                        torch.stack([p.float() for p in predictions], dim=0).mean(dim=0)
                    ]
            return outputs
        finally:
            self._feature_ids = None
            self._capture_features = False
            self._feature_values.clear()

    def close_feature_hooks(self):
        for handle in self._feature_handles:
            handle.remove()


class AttentionGroup:
    def __init__(self, backbone, layers):
        self.captures = [old.SameForwardLayerAttentionCapture(backbone, i) for i in layers]

    @property
    def enabled(self):
        return any(c.enabled for c in self.captures)

    @enabled.setter
    def enabled(self, enabled):
        for capture in self.captures:
            capture.enabled = enabled

    def clear(self):
        for capture in self.captures:
            capture.clear()

    def get(self, index=0):
        return self.captures[index].get()

    def close(self):
        for capture in self.captures:
            capture.close()


def validate_same_forward_attention_capture(model, capture, *args, **kwargs):
    results = [old.validate_same_forward_attention_capture(model, c, *args, **kwargs)
               for c in capture.captures]
    capture.clear()
    return {key: max(r[key] for r in results) for key in results[0]}


def mean_student_attention_bins(attentions, input_ids, attention_mask, level_ids,
                                topk_positions, topk_token_ids, topk_valid_mask):
    """Average raw text rows, NOT normalized per-layer bins; preserve autograd."""
    if not attentions:
        raise ValueError("Student attention layer list is empty")
    batch_size, levels, topk = topk_positions.shape
    if levels != len(level_ids):
        raise ValueError("Attention level count mismatch")
    result = torch.zeros(batch_size, levels, topk + 1, device=input_ids.device, dtype=torch.float32)
    masses = torch.zeros(batch_size, levels, device=input_ids.device, dtype=torch.float32)
    for b in range(batch_size):
        valid_len = int(attention_mask[b].sum().item())
        query_len = valid_len - levels
        if query_len <= 0:
            raise RuntimeError("Student sample has no query tokens")
        for level, token_id in enumerate(level_ids):
            matches = (input_ids[b, :valid_len] == token_id).nonzero(as_tuple=True)[0]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one student level token {token_id}")
            position = int(matches[0].item())
            # Slice before FP32 conversion: do not allocate K full FP32 S x S maps.
            rows = [a[b, :, position, :query_len].float().mean(0).clamp_min(0)
                    for a in attentions]
            raw = torch.stack(rows).mean(0)
            mass = raw.sum()
            masses[b, level] = mass
            scores = raw / mass.clamp_min(1e-12)
            valid = topk_valid_mask[b, level].bool()
            selected = topk_positions[b, level, valid].long()
            if selected.numel():
                if bool(((selected < 0) | (selected >= query_len)).any()):
                    raise RuntimeError("Teacher top-k position is outside the student query")
                if not torch.equal(input_ids[b, selected], topk_token_ids[b, level, valid].long()):
                    raise RuntimeError("Teacher/student tokenization mismatch at attention top-k")
                selected_scores = scores[selected]
                result[b, level, :selected.numel()] = selected_scores
                result[b, level, -1] = (1 - selected_scores.sum()).clamp_min(0)
            else:
                result[b, level, -1] = 1.
    return result / result.sum(-1, keepdim=True).clamp_min(1e-12), masses


class MultiTrainer(old.DistillationTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alignment_plan = self.distill_config["training_compatibility"]["multilayer"]
        if self.attention_capture:
            self.attention_capture.close()
            self.attention_capture = AttentionGroup(self.model.core.backbone,
                                                    self.alignment_plan["attention"]["student_layers"])

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        features = inputs.pop("bank_features", None)
        attention = {k: inputs.pop(f"bank_{k}", None) for k in SPARSE_KEYS}
        label_mask = inputs["label_mask"]
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        weights = (self.feature_kd_weight, self.attention_kd_weight, self.attention_mass_kd_weight)
        # Reuse unchanged supervised/logit losses and exactly one model forward.
        self.feature_kd_weight = self.attention_kd_weight = self.attention_mass_kd_weight = 0.0
        try:
            total, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        finally:
            self.feature_kd_weight, self.attention_kd_weight, self.attention_mass_kd_weight = weights
        losses = {}
        if self.feature_kd_weight > 0:
            if features is None:
                raise RuntimeError("Missing bank feature targets")
            projected = outputs["bank_projected_features"]
            if features.shape[1] != len(projected):
                raise RuntimeError("Feature target/student pair count mismatch")
            values = []
            for i, prediction in enumerate(projected):
                prediction = prediction.float()
                target = features[:, i].float().to(prediction.device)
                level_mask = label_mask.float().to(prediction.device)
                mse = masked_mean((prediction - target).pow(2).mean(-1), level_mask)
                cosine = masked_mean(1 - F.cosine_similarity(prediction, target, dim=-1), level_mask)
                values.append(mse + self.feature_cosine_weight * cosine)
            losses["feature_kd"] = sum(values) / len(values)
            total = total + self.feature_kd_weight * losses["feature_kd"]
        try:
            if self.attention_kd_weight > 0 or self.attention_mass_kd_weight > 0:
                if any(v is None for v in attention.values()):
                    raise RuntimeError("Missing bank attention targets")
                count = attention["topk_positions"].shape[1]
                mean_mode = self.alignment_plan["attention"]["mode"] == "mean"
                if mean_mode and count != 1:
                    raise RuntimeError("Mean attention requires one fused teacher target")
                if not mean_mode and count != len(self.attention_capture.captures):
                    raise RuntimeError("Attention target/student pair count mismatch")
                kl_values, mass_values = [], []
                for i in range(count):
                    if mean_mode and len(self.attention_capture.captures) > 1:
                        bins, mass = mean_student_attention_bins(
                            [self.attention_capture.get(j) for j in range(len(self.attention_capture.captures))],
                            ids, mask, self.level_token_ids, attention["topk_positions"][:, i],
                            attention["topk_token_ids"][:, i], attention["topk_valid_mask"][:, i])
                    else:
                        # Preserve the old single-layer and one-to-one arithmetic.
                        bins, mass = self._attention_bins(self.attention_capture.get(i), ids, mask,
                                                          attention["topk_positions"][:, i],
                                                          attention["topk_token_ids"][:, i],
                                                          attention["topk_valid_mask"][:, i])

                    target = torch.cat([attention["teacher_topk_attention_mean"][:, i].float().to(bins.device),
                                        attention["teacher_other_attention_mean"][:, i].float().to(bins.device)[..., None]], -1)
                    target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
                    level_mask = label_mask.float().to(bins.device)
                    if self.attention_kd_weight > 0:
                        kl_values.append(masked_mean(F.kl_div(bins.clamp_min(1e-8).log(), target, reduction="none").sum(-1), level_mask))
                    if self.attention_mass_kd_weight > 0:
                        target_mass = attention["teacher_text_attention_mass"][:, i].float().to(mass.device)
                        mass_values.append(masked_mean((mass - target_mass).pow(2), level_mask))
                if kl_values:
                    losses["attention_kd"] = sum(kl_values) / count
                    total = total + self.attention_kd_weight * losses["attention_kd"]
                if mass_values:
                    losses["attention_mass_kd"] = sum(mass_values) / count
                    total = total + self.attention_mass_kd_weight * losses["attention_mass_kd"]
        finally:
            if self.attention_capture:
                self.attention_capture.clear()
        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            self._last_distill_losses.update({key: float(value.detach()) for key, value in losses.items()})
        return (total, outputs) if return_outputs else total


def main():
    args = parse_args()
    if args.knowledge_mode == "legacy":
        if (args.feature_alignment != "single" or args.attention_alignment != "single"
                or args.teacher_feature_layers != "original" or args.teacher_attention_layers != "original"
                or args.student_feature_layers is not None or args.student_attention_layers is not None):
            raise ValueError("Multilayer options require --knowledge_mode bank")
        # No new model, hooks, RNG calls or Trainer construction in legacy mode.
        sys.argv = [sys.argv[0], *args._legacy_argv]
        old.main()
        return
    del args._legacy_argv
    run_bank(args)


# run_bank below retains the reference entry's training/evaluation orchestration.


def run_bank(args) -> None:
    validate_loss_args(args)
    if args.gradient_checkpointing and (args.feature_kd_weight > 0 or args.attention_kd_weight > 0 or args.attention_mass_kd_weight > 0):
        raise ValueError("Bank feature/attention capture requires --gradient_checkpointing disabled")
    seed_everything(args.seed)
    attention_enabled = (
        args.attention_kd_weight > 0
        or args.attention_mass_kd_weight > 0
    )
    student_attention_layer = (
        args.student_feature_layer
        if args.student_attention_layer is None
        else args.student_attention_layer
    )

    knowledge_dir = os.path.abspath(args.knowledge_dir)
    manifest_path = os.path.join(knowledge_dir, "manifest.json")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError(
            f"Knowledge manifest is not complete: status={manifest.get('status')}"
        )
    if manifest.get("format") != FORMAT or manifest.get("schema_version") != 1:
        raise ValueError("Bank mode requires per_layer_teacher_bank_v1 knowledge; use legacy mode for original files")
    if manifest.get("dataset") != args.dataset or manifest.get("split") != "train":
        raise ValueError("Knowledge dataset/split mismatch")
    plan = make_plan(args, manifest)
    if (
        args.global_logit_kd_weight > 0
        and not bool(manifest.get("has_trained_global_head", False))
    ):
        raise RuntimeError("Global-logit KD requires a trained teacher global head.")
    if (
        args.local_logit_kd_weight > 0
        and not bool(manifest.get("has_trained_local_heads", False))
    ):
        raise RuntimeError("Local-logit KD requires trained teacher local heads.")

    dataset_cfg = DATASET_ZOO[args.dataset]
    data_dir = args.data_dir or dataset_cfg["path"]
    train_path = os.path.join(data_dir, dataset_cfg["train_file"])
    dev_path = os.path.join(data_dir, dataset_cfg["dev_file"])
    if sha256_file(train_path) != manifest["data_sha256"]:
        raise RuntimeError("Current train JSONL differs from the extracted knowledge source.")

    train_meta = load_preprocess_meta(train_path)
    dev_meta = load_preprocess_meta(dev_path)
    assert_preprocess_meta_compatible(train_meta, dev_meta, "dev")
    num_levels = train_meta["num_levels"]
    level_classes = train_meta["level_classes"]
    total_classes = train_meta["total_classes"]
    level_offsets = build_level_offsets(level_classes)
    if list(manifest["level_classes"]) != level_classes:
        raise RuntimeError("Knowledge level_classes differ from current metadata.")
    if list(manifest["level_offsets"]) != level_offsets:
        raise RuntimeError("Knowledge level_offsets differ from current metadata.")

    required_modules: set[str] = set()
    if args.global_logit_kd_weight > 0 or args.local_logit_kd_weight > 0:
        required_modules.add("logits")
    if args.feature_kd_weight > 0:
        required_modules.add("features")
    if attention_enabled:
        required_modules.add("attention")

    identities = build_sample_identities(
        train_path,
        args.dataset,
        "train",
        num_levels,
    )
    expected_uids = [item["sample_uid"] for item in identities]
    knowledge = BankKnowledge(
        knowledge_dir,
        manifest,
        required_modules,
        expected_uids,
        plan=plan, cache_shards=args.knowledge_cache_shards,
    )

    model_path = args.model_path or MODEL_ZOO[args.model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base_tokenizer_vocab_size = len(tokenizer)
    num_added_level_tokens = add_level_special_tokens(
        tokenizer,
        LEVEL_TOKENS[:num_levels],
    )
    student_level_token_ids = tokenizer.convert_tokens_to_ids(
        LEVEL_TOKENS[:num_levels]
    )
    if attention_enabled:
        student_tokenizer_hash = tokenizer_fingerprint(tokenizer)
        if student_tokenizer_hash != manifest["tokenizer_sha256"]:
            raise RuntimeError(
                "Teacher and student tokenizer vocabularies differ. "
                "Attention KD requires exact tokenizer alignment in this version."
            )

    max_length = int(manifest["max_length"])
    truncate_mode = str(manifest["truncate_mode"])
    train_base = HierarchicalDataset(
        train_path,
        tokenizer,
        num_levels,
        total_classes,
        level_offsets,
        max_length,
        truncate_mode,
        strict=True,
    )
    dev_dataset = HierarchicalDataset(
        dev_path,
        tokenizer,
        num_levels,
        total_classes,
        level_offsets,
        max_length,
        truncate_mode,
        strict=True,
    )
    train_dataset = DistillTrainDataset(train_base, knowledge)

    args.num_levels = num_levels
    backbone = load_student_backbone(
        model_path,
        tokenizer,
        args,
        need_attention=attention_enabled,
    )
    num_student_layers = int(backbone.config.num_hidden_layers)
    if not (
        -num_student_layers
        <= args.student_feature_layer
        < num_student_layers
    ):
        raise ValueError(
            f"--student_feature_layer={args.student_feature_layer} is outside "
            f"[-{num_student_layers}, {num_student_layers})."
        )
    finish_student_plan(plan, args, num_student_layers)
    core = HierarchicalTeacher(
        backbone=backbone,
        tokenizer=tokenizer,
        num_levels=num_levels,
        level_classes=level_classes,
        level_offsets=level_offsets,
        hidden_layer_idx=args.student_feature_layer,
        use_hidden_states=args.student_feature_layer != -1,
        dropout=args.dropout,
    )
    model = MultiStudent(
        core,
        teacher_hidden_size=int(manifest["teacher_hidden_size"]),
        tuning_mode=args.tuning_mode,
        enable_feature_projection=args.feature_kd_weight > 0,
    )
    model.configure_layers(plan)
    if (
        args.local_supervised_bce_weight == 0
        and args.local_logit_kd_weight == 0
    ):
        model.core.level_heads.requires_grad_(False)
    if (
        args.global_supervised_bce_weight == 0
        and args.global_logit_kd_weight == 0
    ):
        model.core.global_head.requires_grad_(False)
    if args.feature_kd_weight == 0:
        model.feature_projection.requires_grad_(False)
    if (
        args.global_supervised_bce_weight == 0
        and args.local_supervised_bce_weight == 0
        and args.global_logit_kd_weight == 0
        and args.local_logit_kd_weight == 0
        and args.feature_kd_weight == 0
    ):
        model.core.post_ln.requires_grad_(False)

    output_dir = args.output_dir or (
        f"/data1/lwq/agt_plus/output_distill/"
        f"{args.dataset}_{Path(model_path).name}_distill_multilayer"
    )
    run_name = args.run_name or Path(output_dir).name
    if args.resume_from_checkpoint is None and os.path.isdir(output_dir) and any(Path(output_dir).iterdir()):
        raise ValueError("Use an empty output_dir for a new bank experiment, or resume explicitly")
    os.makedirs(output_dir, exist_ok=True)
    # Save the exact command used for this run
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        command_parts = []

        for env_name in (
                "CUDA_VISIBLE_DEVICES",
                "SWANLAB_PROJECT",
                "PYTORCH_CUDA_ALLOC_CONF",
        ):
            env_value = os.environ.get(env_name)
            if env_value is not None:
                command_parts.append(f"{env_name}={env_value}")

        command_parts.extend(
            [sys.executable] + [str(arg) for arg in sys.argv]
        )

        with open(
                os.path.join(output_dir, "command.txt"),
                "w",
                encoding="utf-8",
        ) as f:
            f.write(" ".join(command_parts) + "\n")

        write_json(
            os.path.join(output_dir, "args.json"),
            vars(args),
        )
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        sys.stdout = TeeLogger(os.path.join(output_dir, "training.log"))
        sys.stderr = sys.stdout

    selection_specs = {
        "global_macro": ("eval_global_flat_macro_f1", True),
        "local_macro": ("eval_local_flat_macro_f1", True),
        "global_supervised_bce": (
            "eval_global_supervised_bce_loss",
            False,
        ),
        "local_supervised_bce": (
            "eval_local_supervised_bce_loss",
            False,
        ),
    }
    selection_spec = selection_specs.get(args.early_stopping_target)
    selection_metric = selection_spec[0] if selection_spec is not None else None
    selection_greater_is_better = (
        selection_spec[1] if selection_spec is not None else None
    )
    can_use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        data_seed=args.seed,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        bf16=can_use_bf16,
        fp16=not can_use_bf16 and torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        report_to=[] if args.no_swanlab else ["swanlab"],
        run_name=run_name,
        dataloader_pin_memory=True,
        dataloader_num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        metric_for_best_model=selection_metric,
        greater_is_better=selection_greater_is_better,
        load_best_model_at_end=False,
    )

    distill_config = {
        "dataset": args.dataset,
        "student_model_path": model_path,
        "teacher_checkpoint_dir": manifest.get("checkpoint_dir", "fingerprint:" + manifest["teacher_checkpoint_fingerprint"]),
        "knowledge_dir": knowledge_dir,
        "tuning_mode": args.tuning_mode,
        "num_levels": num_levels,
        "level_classes": level_classes,
        "level_offsets": level_offsets,
        "level_tokens": LEVEL_TOKENS[:num_levels],
        "student_level_token_ids": student_level_token_ids,
        "teacher_level_token_ids": manifest["teacher_level_token_ids"],
        "max_length": max_length,
        "truncate_mode": truncate_mode,
        "feature_type": manifest["feature_type"],
        "attention_topk": manifest["attention_topk"],
        "fixed_threshold": args.fixed_threshold,
        "loss_weights": {
            "global_supervised_bce": args.global_supervised_bce_weight,
            "local_supervised_bce": args.local_supervised_bce_weight,
            "global_logit_kd": args.global_logit_kd_weight,
            "local_logit_kd": args.local_logit_kd_weight,
            "feature_kd": args.feature_kd_weight,
            "feature_cosine": args.feature_cosine_weight,
            "attention_kd": args.attention_kd_weight,
            "attention_mass_kd": args.attention_mass_kd_weight,
        },
        "student_feature_layer": args.student_feature_layer,
        "student_attention_layer": student_attention_layer,
        "teacher_feature_layer": manifest.get("feature_layer"),
        "teacher_attention_layer": manifest.get("attention_layer"),
        "feature_kd_alignment": "post_ln_linear_projection",
        "student_attention_capture": "same_forward_selected_layer",
        "attention_mass_loss": "masked_mse",
        "temperature": args.temperature,
        "seed": args.seed,
        "training_compatibility": {
            "multilayer": plan,
            "train_batch": args.train_batch,
            "gradient_accumulation_steps": args.grad_acc,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "early_stopping_target": args.early_stopping_target,
        "early_stopping_metric": selection_metric,
    }
    distill_config["evaluation_compatibility"] = {
        "dataset": args.dataset,
        "model_path": model_path,
        "run_name": run_name,
        "num_levels": num_levels,
        "level_classes": level_classes,
        "level_offsets": level_offsets,
        "label_space_source": train_meta["meta"].get("label_space_source"),
        "layer_label2id": normalize_layer_label2id(train_meta["meta"]),
        "level_tokens": LEVEL_TOKENS[:num_levels],
        "level_token_ids": student_level_token_ids,
        "base_tokenizer_vocab_size": base_tokenizer_vocab_size,
        "tokenizer_vocab_size": len(tokenizer),
        "num_added_level_tokens": num_added_level_tokens,
        "level_tokens_are_new": bool(
            num_added_level_tokens == num_levels
            and all(
                token_id >= base_tokenizer_vocab_size
                for token_id in student_level_token_ids
            )
        ),
        "embedding_training": (
            "trainable_token_indices"
            if args.tuning_mode == "lora"
            else "full_finetuning"
        ),
        "hidden_layer_idx": args.student_feature_layer,
        "hidden_layer_indexing": "transformer_block_0_based",
        "hidden_states_index": (
            args.student_feature_layer
            if args.student_feature_layer < 0
            else args.student_feature_layer + 1
        ),
        "use_hidden_states": args.student_feature_layer != -1,
        "max_length": max_length,
        "truncate_mode": truncate_mode,
        "level_loss_weights": [1.0] * num_levels,
        "local_loss_weight": float(
            args.local_supervised_bce_weight + args.local_logit_kd_weight
        ),
        "global_loss_weight": float(
            args.global_supervised_bce_weight + args.global_logit_kd_weight
        ),
        "early_stop_target": args.early_stopping_target,
        "early_stop_metric": selection_metric,
        "fixed_threshold": args.fixed_threshold,
        "seed": args.seed,
        "data_seed": args.seed,
        "has_trained_local_heads": bool(
            args.local_supervised_bce_weight > 0
            or args.local_logit_kd_weight > 0
        ),
        "has_trained_global_head": bool(
            args.global_supervised_bce_weight > 0
            or args.global_logit_kd_weight > 0
        ),
        "checkpoint_format": args.tuning_mode,
    }

    trainer = MultiTrainer(
        tokenizer_for_save=tokenizer,
        distill_config=distill_config,
        level_token_ids=student_level_token_ids,
        level_classes=level_classes,
        level_offsets=level_offsets,
        global_supervised_bce_weight=args.global_supervised_bce_weight,
        local_supervised_bce_weight=args.local_supervised_bce_weight,
        global_logit_kd_weight=args.global_logit_kd_weight,
        local_logit_kd_weight=args.local_logit_kd_weight,
        feature_kd_weight=args.feature_kd_weight,
        feature_cosine_weight=args.feature_cosine_weight,
        attention_kd_weight=args.attention_kd_weight,
        attention_mass_kd_weight=args.attention_mass_kd_weight,
        student_attention_layer=student_attention_layer,
        temperature=args.temperature,
        fixed_threshold=args.fixed_threshold,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_target=args.early_stopping_target,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=stack_collator,
    )
    if trainer.attention_capture is not None:
        validation_sample = train_base[0]
        validation_device = next(trainer.model.parameters()).device
        attention_validation = validate_same_forward_attention_capture(
            trainer.model,
            trainer.attention_capture,
            validation_sample["input_ids"].unsqueeze(0).to(validation_device),
            validation_sample["attention_mask"].unsqueeze(0).to(
                validation_device
            ),
            num_levels,
        )
        if trainer.is_world_process_zero():
            print(
                "Student same-forward attention capture validation passed: "
                f"seq_len={int(attention_validation['validation_seq_len'])}, "
                f"max_row_sum_error="
                f"{attention_validation['max_row_sum_error']:.8f}, "
                f"mean_row_sum_error="
                f"{attention_validation['mean_row_sum_error']:.8f}"
            )
    resume_from_checkpoint: str | None = None
    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint.lower() == "latest":
            resume_from_checkpoint = get_last_checkpoint(output_dir)
            if resume_from_checkpoint is None:
                raise FileNotFoundError(
                    f"No checkpoint-* directory found under {output_dir}."
                )
        else:
            resume_from_checkpoint = os.path.abspath(
                args.resume_from_checkpoint
            )
            if not os.path.isdir(resume_from_checkpoint):
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {resume_from_checkpoint}"
                )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(os.path.join(output_dir, "final_model"))
    final_metrics = trainer.evaluate(
        eval_dataset=dev_dataset,
        metric_key_prefix="final_eval",
    )
    if trainer.is_world_process_zero():
        write_json(
            os.path.join(output_dir, "final_eval_metrics.json"),
            final_metrics,
        )
    if trainer.attention_capture is not None:
        trainer.attention_capture.close()
    model.close_feature_hooks()


if __name__ == "__main__":
    main()
