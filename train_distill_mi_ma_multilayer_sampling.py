"""Layer-selection entry point; all KD arithmetic stays in the existing module.

See MULTILAYER_SAMPLING.md. Layer indices in saved schedules are zero-based.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys

import torch
from transformers import TrainerCallback
import train_distill_mi_ma_multilayer as base


def choose_layers(candidates, size, strategy, seed, epoch):
    candidates = sorted(candidates)
    if not 1 <= size <= len(candidates) or len(set(candidates)) != len(candidates):
        raise ValueError("Need distinct teacher candidates and 1 <= student count <= teacher count")
    if strategy == "uniform":
        # ceil(m*N/M)-1, exact integer arithmetic; m starts at 1.
        return [candidates[(m * len(candidates) + size - 1) // size - 1]
                for m in range(1, size + 1)]
    if strategy == "random_epoch":
        return sorted(random.Random(seed + epoch).sample(candidates, size))
    raise ValueError(f"Unknown selection strategy: {strategy}")


def build_schedule(plan, epochs):
    policy = plan["layer_sampling"]
    mappings = []
    for epoch in range(math.ceil(epochs)):
        row = {"epoch_index": epoch}
        for kind in ("feature", "attention"):
            entry = plan[kind]
            if entry:
                row[kind] = {
                    "teacher_layers": choose_layers(
                        policy["teacher_candidates"], len(entry["student_layers"]),
                        policy["strategy"], policy["seed"], epoch),
                    "student_layers": list(entry["student_layers"]),
                }
        mappings.append(row)
    return {"schema_version": 1, "indexing": "zero_based_decoder_blocks",
            "policy": copy.deepcopy(policy), "epochs": mappings}


def save_or_check(path, value, write=True):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Layer schedule mismatch: {path}")
    elif write:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)


class LayerScheduleCallback(TrainerCallback):
    def __init__(self, knowledge, schedule):
        self.knowledge = knowledge
        self.schedule = schedule
        self.epoch_index = 0

    def on_train_begin(self, args, state, control, **kwargs):
        # Trainer restores state.epoch before this event, including fractional epochs.
        self.epoch_index = math.floor(state.epoch or 0)
        save_or_check(Path(args.output_dir) / "layer_schedule.json", self.schedule,
                      write=state.is_world_process_zero)

    def on_epoch_begin(self, args, state, control, **kwargs):
        row = copy.deepcopy(self.schedule["epochs"][self.epoch_index])
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            payload = [row if torch.distributed.get_rank() == 0 else None]
            torch.distributed.broadcast_object_list(payload, src=0)
            if payload[0] != row:
                raise RuntimeError("Layer schedules differ across distributed ranks")
            row = payload[0]
        # num_workers=0: no worker copies/prefetched targets survive the transition.
        self.knowledge.cache.clear()
        for kind in ("feature", "attention"):
            if kind in row:
                self.knowledge.plan[kind]["teacher_layers"] = row[kind]["teacher_layers"]
        if state.is_world_process_zero:
            print("Layer mapping: " + json.dumps(row))

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_index += 1

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            save_or_check(Path(args.output_dir) / f"checkpoint-{state.global_step}" /
                          "layer_schedule.json", self.schedule)


def run_sampling(args, strategy):
    if args.knowledge_mode != "bank":
        raise ValueError("Layer sampling requires --knowledge_mode bank")
    if strategy == "random_epoch" and args.num_workers != 0:
        raise ValueError("random_epoch requires --num_workers 0 to prevent stale prefetched layer targets")
    enabled = [kind for kind, active in (
        ("feature", args.feature_kd_weight > 0),
        ("attention", args.attention_kd_weight > 0 or args.attention_mass_kd_weight > 0)) if active]
    if not enabled:
        raise ValueError("Sampling requires a nonzero feature or attention KD weight; use fixed for logits only")
    for kind in enabled:
        if getattr(args, f"{kind}_alignment") != "one_to_one":
            raise ValueError(f"Sampling requires --{kind}_alignment one_to_one")
        if getattr(args, f"teacher_{kind}_layers") != "all" or getattr(args, f"student_{kind}_layers") != "all":
            raise ValueError(f"Sampling requires --teacher_{kind}_layers all --student_{kind}_layers all")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    original_make = base.make_plan
    original_finish = base.finish_student_plan
    original_bank = base.BankKnowledge
    original_trainer = base.MultiTrainer
    context = {}

    def make_plan(args, manifest):
        # Validate ALL candidate shards before training, not just epoch zero's subset.
        plan = original_make(args, manifest)
        candidates = list(range(manifest["num_decoder_layers"]))
        if sorted(manifest["layer_ids"]) != candidates:
            raise ValueError("Sampling requires a complete --layers all teacher bank")
        plan["layer_sampling"] = {
            "strategy": strategy, "seed": args.layer_selection_seed,
            "teacher_candidates": candidates, "student_selection": "all",
            "uniform_rule": "ceil(m*N/M)-1; m=1..M",
            "random_rule": "sorted(Random(seed+epoch_index).sample(candidates,M))",
            "shared_feature_attention_mapping": True,
            "planned_epochs": args.epochs,
        }
        return plan

    class SamplingKnowledge(original_bank):
        def __init__(self, *pos, **kw):
            super().__init__(*pos, **kw)
            context["knowledge"] = self

    def finish_plan(plan, args, count):
        for kind in enabled:
            plan[kind]["teacher_layers"] = choose_layers(
                plan["layer_sampling"]["teacher_candidates"], count,
                strategy, args.layer_selection_seed, 0)
        original_finish(plan, args, count)
        # Only the loader's private copy changes each epoch. Checkpoint compatibility
        # metadata, student hooks and the trainer's loss plan remain immutable.
        context["knowledge"].plan = copy.deepcopy(plan)
        context["schedule"] = build_schedule(plan, args.epochs)
        return plan

    class SamplingTrainer(original_trainer):
        def __init__(self, *pos, **kw):
            super().__init__(*pos, **kw)
            if strategy == "random_epoch":
                self.add_callback(LayerScheduleCallback(context["knowledge"], context["schedule"]))

        def train(self, resume_from_checkpoint=None, **kw):
            schedule = context["schedule"]
            if resume_from_checkpoint:
                path = Path(resume_from_checkpoint) / "layer_schedule.json"
                if not path.is_file():
                    raise ValueError(f"Sampling checkpoint lacks layer schedule: {path}")
                save_or_check(path, schedule, write=False)
            save_or_check(Path(self.args.output_dir) / "layer_schedule.json", schedule,
                          write=self.is_world_process_zero())
            return super().train(resume_from_checkpoint=resume_from_checkpoint, **kw)

        def save_model(self, output_dir=None, _internal_call=False):
            super().save_model(output_dir=output_dir, _internal_call=_internal_call)
            # Include the schedule in checkpoints, best models and final_model.
            if self.is_world_process_zero():
                save_or_check(Path(output_dir or self.args.output_dir) / "layer_schedule.json",
                              context["schedule"])

    # Scoped dependency overrides let the ORIGINAL run_bank orchestrate the entire
    # run. No duplicate model, optimizer, loss, evaluation or early-stopping code.
    try:
        base.make_plan = make_plan
        base.finish_student_plan = finish_plan
        base.BankKnowledge = SamplingKnowledge
        base.MultiTrainer = SamplingTrainer
        base.run_bank(args)
    finally:
        base.make_plan = original_make
        base.finish_student_plan = original_finish
        base.BankKnowledge = original_bank
        base.MultiTrainer = original_trainer


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--layer_selection", choices=("fixed", "uniform", "random_epoch"), default="fixed")
    extra, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        parser.print_help()
    argv = sys.argv
    try:
        sys.argv = [argv[0], *remaining]
        if extra.layer_selection == "fixed":
            # Exact original entry, including legacy and every single/mean/pair mode.
            base.main()
            return
        args = base.parse_args()
    finally:
        sys.argv = argv
    del args._legacy_argv
    args.layer_selection = extra.layer_selection
    run_sampling(args, extra.layer_selection)


if __name__ == "__main__":
    main()
