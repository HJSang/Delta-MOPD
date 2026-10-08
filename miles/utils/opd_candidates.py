"""Wire schema for unreduced Open-MOPD candidate rewards (response, candidate)."""

OPD_CANDIDATE_DTYPES = {
    "opd_candidate_ids": "int64",
    "opd_candidate_old_log_probs": "float32",
    "opd_candidate_rewards": "float32",
    "opd_candidate_mask": "bool",
}
