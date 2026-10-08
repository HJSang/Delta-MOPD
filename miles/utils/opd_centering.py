"""Explicit sampled-token centering, without variance/support normalization."""

import torch

from miles.utils.paper_opd_config import paper_opd_enabled


def opd_centering_mode(args):
    if paper_opd_enabled(args):
        return "batch"
    mode = getattr(args, "opd_reward_centering", None)
    if mode is None:  # Preserve historical defaults for old configurations.
        return "response" if getattr(args, "opd_target_mode", "endpoint") == "shiftmopd" else "none"
    if mode not in ("none", "response"):
        raise ValueError(f"Unknown OPD reward centering mode: {mode}")
    return mode


def center_sampled_rewards(rewards, loss_mask=None, *, center=True):
    """Return masked detached rewards and their valid sampled-token baseline.

    One invocation is one response (original workspace microbatch-size-one
    convention), NOT a candidate-support mean or a distributed batch mean.
    Empty/fully masked responses return zeros and a zero baseline.
    """
    rewards = torch.as_tensor(rewards).detach()
    if rewards.ndim != 1:
        raise ValueError("OPD centering expects one-dimensional sampled rewards.")
    mask = torch.ones_like(rewards, dtype=torch.bool) if loss_mask is None else torch.as_tensor(
        loss_mask, device=rewards.device, dtype=torch.bool
    )
    if mask.shape != rewards.shape:
        raise ValueError("OPD centering response/loss-mask shape mismatch.")
    if not torch.isfinite(rewards[mask]).all():
        raise ValueError("Non-finite valid sampled OPD reward.")
    baseline = rewards[mask].mean() if center and mask.any() else rewards.new_zeros(())
    return torch.where(mask, rewards - baseline, 0.0), baseline
