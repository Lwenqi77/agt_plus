"""Explicit training-time layer selection and fusion; no teacher retraining.

Feature inputs are the stored post_ln native-layer representations. Averaging
these is an experimental target, with no extra backbone norm or post_ln.
"""
from __future__ import annotations

import random

import torch


def select_layers(available, mode="last", k=4, fixed=None, seed=42):
    available = sorted(available)
    if not available or len(set(available)) != len(available):
        raise ValueError("Candidate layer IDs must be non-empty and unique")
    if mode == "fixed":
        selected = sorted(fixed or [])
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(available):
            raise ValueError("Fixed layer IDs must be unique members of the bank")
        return selected
    if not 1 <= k <= len(available):
        raise ValueError("k must be within candidate count")
    if mode == "last":
        return available[-k:]
    if mode == "random":
        return sorted(random.Random(seed).sample(available, k))
    raise ValueError(f"Unknown selection mode: {mode}")


def pair_layers(teacher_layers, student_layers):
    if (not teacher_layers or len(teacher_layers) != len(student_layers)
            or len(set(teacher_layers)) != len(teacher_layers)
            or len(set(student_layers)) != len(student_layers)):
        raise ValueError("One-to-one mapping requires equal nonzero counts and unique layers")
    return list(zip(sorted(teacher_layers), sorted(student_layers)))


def mean_features(features):
    """Input [layers,batch,levels,hidden]; FP32 accumulation, no extra norm."""
    if features.ndim != 4 or features.shape[0] < 1:
        raise ValueError("Expected [layers,batch,levels,hidden]")
    return features.float().mean(dim=0)


def mean_attention(raw_text_attention, query_lengths):
    """Average raw layer weights, then normalize on valid text positions.

    Input [layers,batch,levels,text]; returns (distribution, pre-normalization mass).
    Only combine aligned query_token_ids/offsets from the same sample identities.
    Apply top-K AFTER this operation, never before it.
    """
    if raw_text_attention.ndim != 4 or raw_text_attention.shape[0] < 1:
        raise ValueError("Expected [layers,batch,levels,text]")
    scores = raw_text_attention.float().mean(dim=0)
    lengths = query_lengths.to(scores.device)
    if lengths.shape != (scores.shape[0],) or bool(((lengths <= 0) | (lengths > scores.shape[-1])).any()):
        raise ValueError("Invalid query lengths")
    valid = torch.arange(scores.shape[-1], device=scores.device)[None, :] < lengths[:, None]
    scores = scores.masked_fill(~valid[:, None, :], 0)
    mass = scores.sum(dim=-1)
    if not bool(torch.isfinite(scores).all()) or bool((scores < 0).any()) or bool((mass <= 0).any()):
        raise ValueError("Invalid attention distribution")
    return scores / mass[..., None], mass
