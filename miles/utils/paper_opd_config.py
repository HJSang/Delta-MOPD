"""Fail-closed configuration contract for the final-draft sampled-token path."""


def paper_opd_enabled(args) -> bool:
    return getattr(args, "opd_objective", "legacy") == "paper"


def validate_paper_opd_args(args) -> None:
    if not paper_opd_enabled(args):
        return
    required = {
        "use_opd": True,
        "opd_type": "sglang",
        "loss_type": "policy_loss",
        "advantage_estimator": "grpo",
        "rollout_temperature": 1.0,
        "rollout_top_p": 1.0,
        "rollout_min_p": 0.0,
        "opd_kl_coef": 1.0,
    }
    for name, expected in required.items():
        if getattr(args, name, None) != expected:
            raise ValueError(f"Paper OPD requires --{name.replace('_', '-')}={expected}.")
    disabled = (
        "normalize_advantages",
        "use_kl_loss",
        "use_tis",
        "use_opsm",
        "partial_rollout",
        "fully_async",
        "use_dynamic_global_batch_size",
        "calculate_per_token_loss",
        "get_mismatch_metrics",
        "true_on_policy_mode",
    )
    for name in disabled:
        if getattr(args, name, False):
            raise ValueError(f"Paper OPD does not support --{name.replace('_', '-')}.")
    for name in ("entropy_coef", "kl_coef", "hidden_dropout", "attention_dropout"):
        if getattr(args, name, 0) != 0:
            raise ValueError(f"Paper OPD requires --{name.replace('_', '-')}=0.")
    if getattr(args, "rollout_top_k", -1) not in (-1, 0):
        raise ValueError("Paper OPD disables generation top-k (partition top-k is a separate choice).")
    if getattr(args, "opd_reward_centering", None) not in (None, "batch"):
        raise ValueError("Paper OPD requires optimization-batch centering, not response/none.")
    if getattr(args, "opd_teacher_selection", None) not in ("all", "routed"):
        raise ValueError("Paper OPD requires an explicit --opd-teacher-selection all|routed in BOTH arms.")
    if not getattr(args, "opd_anchor_url", None) or not getattr(args, "opd_teacher_urls", None):
        raise ValueError("Paper OPD requires the frozen initial-student anchor and named teacher URLs.")
    if args.opd_effective_vocab_size <= 0 or args.opd_topk_fallback_chunk_size <= 0:
        raise ValueError("Paper OPD vocabulary and scoring chunk sizes must be positive.")
    if args.opd_log_prob_top_k > args.opd_effective_vocab_size:
        raise ValueError("Student top-k cannot exceed the effective vocabulary.")
    if args.opd_top_k_strategy != "only-student":
        raise ValueError("Paper restricted support uses student top-k, never a candidate objective.")
    if args.opd_paper_partition == "student-topk" and args.opd_log_prob_top_k <= 0:
        raise ValueError("Restricted partition requires --opd-log-prob-top-k > 0.")
    if args.opd_paper_partition == "full" and args.opd_log_prob_top_k != 0:
        raise ValueError("Full partition does not need rollout top-k; set --opd-log-prob-top-k=0.")
    if getattr(args, "num_steps_per_rollout", None) not in (None, 1):
        raise ValueError("Paper OPD currently supports one synchronous update per rollout.")
    if getattr(args, "update_weights_interval", 1) != 1:
        raise ValueError("Paper OPD must publish fresh student weights after every update.")
    if getattr(args, "custom_generate_function_path", None):
        raise ValueError("Paper OPD uses the standard vocabulary-masked student generator.")
    if getattr(args, "custom_convert_samples_to_train_data_path", None):
        raise ValueError("Paper OPD requires the standard full-batch train-data conversion.")
    for option, suffix in (
        ("custom_rm_path", "reward_func"),
        ("custom_reward_post_process_path", "post_process_rewards"),
    ):
        paths = {f"miles.rollout.{module}.{suffix}" for module in ("paper_opd", "on_policy_distillation")}
        if getattr(args, option, None) not in paths:
            raise ValueError(f"Paper OPD requires its {option} hook; expected one of {sorted(paths)}.")
    if getattr(args, "rollout_function_path", None) not in (None, "miles.rollout.sglang_rollout.generate_rollout"):
        raise ValueError("Paper OPD currently requires the standard legacy SGLang rollout with vocabulary masking.")
    for name in ("tensor_model_parallel_size", "context_parallel_size"):
        if getattr(args, name, 1) != 1:
            raise ValueError(
                "Paper loss currently supports training TP=CP=1; DP and microbatch accumulation are supported."
            )


def effective_vocab_bias(effective_vocab_size: int, model_vocab_size: int) -> dict[str, float]:
    """Finite wire values whose exp underflows to zero in the SGLang sampler."""
    if not 0 < effective_vocab_size <= model_vocab_size:
        raise ValueError("Effective vocabulary must fit the student's model logits.")
    return {str(token): -1e30 for token in range(effective_vocab_size, model_vocab_size)}
