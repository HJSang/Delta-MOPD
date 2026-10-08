"""Compatibility exports for legacy ``miles.rollout.deepscaler`` imports.

The reward implementations live in :mod:`miles.rollout.rm_hub.deepscaler`,
but some rollout modules still import them from this historical path.
"""

from .rm_hub.deepscaler import get_deepscaler_rule_based_reward, get_gemma_math_reward

__all__ = ["get_deepscaler_rule_based_reward", "get_gemma_math_reward"]
