"""Feature-carrier ablation; reuse the user's original training and loss code.

--feature_representation level_token: unchanged original experiment (default).
--feature_representation text_mean: pool valid text tokens for Feature KD only.
See FEATURE_ABLATION.md. Classification still uses hierarchy-level tokens.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import os
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

import train_distill_mi_ma as old
import extract_teacher_text_features as generic


class TextFeatureReader:
    def __init__(self, root, reference, expected_uids, cache_shards=2):
        self.root = os.path.abspath(root)
        self.manifest = old.read_json(os.path.join(self.root,"manifest.json"))
        m = self.manifest
        if m.get("format") != generic.FORMAT or m.get("schema_version") != 1 or m.get("status") != "complete":
            raise ValueError("Generic feature knowledge has incompatible format or is incomplete")
        if m.get("feature_type") != generic.FEATURE_TYPE:
            raise ValueError("Unsupported text feature definition")
        # Match the original teacher/classification layer, data and token semantics.
        for key in ("dataset","split","data_sha256","teacher_checkpoint_fingerprint","tokenizer_sha256",
                    "max_length","truncate_mode","num_levels","level_classes","level_offsets",
                    "teacher_level_token_ids","teacher_hidden_size","feature_layer","num_samples"):
            if key not in reference or m.get(key) != reference[key]:
                raise ValueError(f"Generic and original knowledge differ: {key}")
        if m["num_samples"] != len(expected_uids) or cache_shards < 1 or m["shard_size"] < 1:
            raise ValueError("Generic sample count or cache/shard size is invalid")
        if m["num_shards"] != (m["num_samples"]+m["shard_size"]-1)//m["shard_size"]:
            raise ValueError("Generic shard count mismatch")
        self.cache, self.cache_limit = OrderedDict(), cache_shards
        for shard in range(m["num_shards"]):
            start,end = shard*m["shard_size"],min(m["num_samples"],(shard+1)*m["shard_size"])
            uids = old.uid_hex_to_tensor(expected_uids[start:end])
            if not generic.valid_shard(self.root,shard,start,end,uids,m):
                raise RuntimeError(f"Generic feature shard failed UID/checksum/schema validation: {shard}")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["cache"] = OrderedDict()
        return state

    def item(self, idx):
        if not 0 <= idx < self.manifest["num_samples"]:
            raise IndexError(idx)
        shard,row = divmod(idx,self.manifest["shard_size"])
        if shard not in self.cache:
            while len(self.cache) >= self.cache_limit:
                self.cache.popitem(last=False)
            self.cache[shard] = load_file(generic.paths(self.root,shard)[0],device="cpu")
        self.cache.move_to_end(shard)
        values = self.cache[shard]
        return values["teacher_text_features"][row],values["input_ids"][row],int(values["valid_lengths"][row])

    def compatibility(self):
        return {"representation":"text_mean","generic_bank_fingerprint":self.manifest["bank_fingerprint"],
                "pooling":generic.FEATURE_TYPE,"teacher_feature_layer":self.manifest["feature_layer"],
                "storage_dtype":self.manifest["storage_dtype"],
                "loss":"original_mse_plus_cosine_broadcast_original_label_mask_v1"}


class TextMeanKnowledge(old.OfflineKnowledge):
    def __init__(self, knowledge_dir, manifest, required_modules, expected_uids, *, generic_dir, cache_shards):
        if manifest.get("format") is not None:
            raise ValueError("This ablation uses the original single-layer --knowledge_dir (teacher_knowledge_train), not a multilayer bank")
        super().__init__(knowledge_dir,manifest,set(required_modules)-{"features"},expected_uids)
        self.text_reader = TextFeatureReader(generic_dir,manifest,expected_uids,cache_shards)

    def item(self, idx):
        value = super().item(idx)
        feature,tokens,length = self.text_reader.item(idx)
        # Repetition only adapts the old loss interface; this is ONE generic vector,
        # not newly trained level-specific vectors. Keeps original sample weights.
        value["teacher_level_features"] = feature[None].expand(self.manifest["num_levels"],-1)
        value["generic_input_ids"] = tokens
        value["generic_valid_length"] = length
        return value


class TextMeanDataset(old.DistillTrainDataset):
    def __getitem__(self, idx):
        value = super().__getitem__(idx)
        tokens = value.pop("generic_input_ids")
        length = value.pop("generic_valid_length")
        mask = value["attention_mask"]
        if (int(mask.sum()) != length or not bool((mask[:length] == 1).all()) or bool(mask[length:].any())
                or not torch.equal(tokens[:length].long(),value["input_ids"][:length].long())):
            raise ValueError(f"Generic teacher/student token mismatch at dataset_idx={idx}")
        return value


class TextMeanStudent(old.DistillStudent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self._text_context = None
        self._text_feature = None
        self._text_handle = self.core.backbone.register_forward_hook(self._capture_text)

    def _capture_text(self, module, args, output):
        if self._text_context is None:
            return
        ids,mask = self._text_context
        self._text_feature = generic.text_mean_post_ln(
            generic.selected_hidden(self.core,output),ids,mask,self.core.level_token_ids,self.core.post_ln)

    def forward(self, input_ids, attention_mask, **kwargs):
        self._text_context = (input_ids,attention_mask) if self.enable_feature_projection else None
        self._text_feature = None
        try:
            outputs = super().forward(input_ids,attention_mask,**kwargs)
            if self.enable_feature_projection:
                if self._text_feature is None:
                    raise RuntimeError("Student text feature hook did not capture the selected layer")
                projected = self.feature_projection(self._text_feature)
                outputs["projected_level_features"] = projected[:,None,:].expand(-1,self.core.num_levels,-1)
            return outputs
        finally:
            self._text_context = None
            self._text_feature = None

    def close_text_hook(self):
        self._text_handle.remove()


def run_text_mean(args):
    if not args.generic_feature_dir:
        raise ValueError("text_mean with feature KD enabled requires --generic_feature_dir")
    if args.output_dir is None:
        model_path = args.model_path or old.MODEL_ZOO[args.model_key]
        args.output_dir = f"/data1/lwq/agt_plus/output_distill/{args.dataset}_{Path(model_path).name}_distill_text_mean"
    if args.resume_from_checkpoint is None and os.path.isdir(args.output_dir) and any(Path(args.output_dir).iterdir()):
        raise ValueError("Use an empty output_dir for the new feature ablation, or resume explicitly")
    original = {name:getattr(old,name) for name in
                ("parse_args","OfflineKnowledge","DistillTrainDataset","DistillStudent","DistillationTrainer")}
    context = {}

    def knowledge_factory(*pos,**kw):
        knowledge = TextMeanKnowledge(*pos,**kw,generic_dir=args.generic_feature_dir,
                                      cache_shards=args.generic_feature_cache_shards)
        context["knowledge"] = knowledge
        return knowledge

    def trainer_factory(*pos,**kw):
        config = kw["distill_config"]
        reader = context["knowledge"].text_reader
        config["generic_feature_dir"] = reader.root
        config["feature_representation"] = "text_mean"
        config["feature_type"] = generic.FEATURE_TYPE
        config["feature_kd_alignment"] = "text_mean_then_post_ln_shared_original_linear"
        config["training_compatibility"]["feature_ablation"] = reader.compatibility()
        context["model"] = kw["model"]
        print("Feature KD: text-token mean -> original post_ln -> shared projection; "
              "original MSE + cosine and label_mask reduction. Classification still uses level tokens.")
        # Original Trainer, with entirely unchanged compute_loss/evaluate/save/restore.
        return original["DistillationTrainer"](*pos,**kw)

    try:
        old.parse_args = lambda: args
        old.OfflineKnowledge = knowledge_factory
        old.DistillTrainDataset = TextMeanDataset
        old.DistillStudent = TextMeanStudent
        old.DistillationTrainer = trainer_factory
        old.main()
    finally:
        if "model" in context:
            context["model"].close_text_hook()
        for name,value in original.items():
            setattr(old,name,value)


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--feature_representation",choices=("level_token","text_mean"),default="level_token")
    p.add_argument("--generic_feature_dir",default=None)
    p.add_argument("--generic_feature_cache_shards",type=int,default=2)
    extra,remaining = p.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        p.print_help()
    argv = sys.argv
    try:
        sys.argv = [argv[0],*remaining]
        args = old.parse_args()
    finally:
        sys.argv = argv
    vars(args).update(vars(extra))
    if args.feature_representation == "level_token" or args.feature_kd_weight == 0:
        parser = old.parse_args
        try:
            old.parse_args = lambda: args
            old.main()
        finally:
            old.parse_args = parser
    else:
        run_text_mean(args)


if __name__ == "__main__":
    main()
