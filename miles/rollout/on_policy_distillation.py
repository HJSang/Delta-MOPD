import asyncio
import math
from argparse import Namespace
from collections.abc import Iterable
from typing import Any
from time import perf_counter

import aiohttp
import torch

from miles.utils.lora import LORA_ADAPTER_NAME, lora_rollout_enabled
from miles.utils.opd_centering import center_sampled_rewards, opd_centering_mode
from miles.utils.paper_opd_config import paper_opd_enabled
from miles.utils.types import Sample
from miles.rollout.shift_composition import compose_reverse_kl, compose_scorer_responses
from miles.utils.selected_logprobs import decode_score_response, selected_logprob_rows
from miles.utils.selected_logprob_transport import DECODED_FIELD, FIELD as PACKED_SCORES_FIELD, REQUEST_FIELD

TopLogprobs = list[list[Any]]
LogprobMaps = list[dict[int, float]]

TOP_K_STRATEGIES = {"only-student", "only-teacher", "intersection", "union", "xor"}
REWARD_WEIGHT_MODES = {"student_p", "teacher_p", "none"}
OPD_TARGET_MODES = {"endpoint", "shiftmopd"}

STUDENT_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-teacher"}
TEACHER_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-student"}
TEACHER_ON_STUDENT_STRATEGIES = {"only-student", "union", "xor"}
STUDENT_ON_TEACHER_STRATEGIES = {"only-teacher", "union", "xor"}

# Reserved teacher name in --opd-teacher-urls used as the fallback route.
DEFAULT_TEACHER_NAME = "default"


def parse_teacher_urls(values: Iterable[str] | None) -> dict[str, str]:
    """Parse ``NAME=URL`` entries from ``--opd-teacher-urls`` into a routing map.

    Splits on the first ``=`` only, so URLs containing ``=`` (e.g. query
    strings) survive intact. Raises on malformed entries and duplicate names
    so misconfiguration fails at startup, not mid-rollout.
    """
    url_map: dict[str, str] = {}
    for value in values or []:
        name, sep, url = value.partition("=")
        name, url = name.strip(), url.strip()
        if not sep or not name or not url:
            raise ValueError(f"Invalid --opd-teacher-urls entry {value!r}; expected NAME=URL.")
        if name in url_map:
            raise ValueError(f"Duplicate teacher name {name!r} in --opd-teacher-urls.")
        url_map[name] = url
    return url_map


def _teacher_route_for_sample(args: Namespace, sample: Sample) -> tuple[str, str]:
    """Resolve the teacher scoring endpoint for one sample.

    Without ``--opd-teacher-urls`` every sample goes to ``--rm-url`` (the
    original single-teacher path, unchanged). With it, the sample is routed by
    the teacher name in ``sample.metadata[--opd-teacher-key]``; samples whose
    name is missing or unknown fall back to the reserved ``default`` entry,
    and raise if no default is configured — silently distilling from the
    wrong teacher is worse than failing the rollout.
    """
    url_map = parse_teacher_urls(getattr(args, "opd_teacher_urls", None))
    if not url_map:
        return DEFAULT_TEACHER_NAME, args.rm_url

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    key = getattr(args, "opd_teacher_key", "opd_teacher")
    name = metadata.get(key)
    if name is not None:
        url = url_map.get(str(name))
        if url is not None:
            return str(name), url
        if DEFAULT_TEACHER_NAME in url_map:
            return DEFAULT_TEACHER_NAME, url_map[DEFAULT_TEACHER_NAME]
        raise ValueError(
            f"Sample metadata[{key!r}]={name!r} matches no --opd-teacher-urls name "
            f"(known: {sorted(url_map)}) and no 'default' entry is configured."
        )
    if DEFAULT_TEACHER_NAME in url_map:
        return DEFAULT_TEACHER_NAME, url_map[DEFAULT_TEACHER_NAME]
    raise ValueError(f"Sample metadata is missing teacher key {key!r} and --opd-teacher-urls has no 'default' entry.")


def _teacher_url_for_sample(args: Namespace, sample: Sample) -> str:
    return _teacher_route_for_sample(args, sample)[1]


def _adapter_map(values: Iterable[str] | None, option: str) -> dict[str, str | None]:
    result = {}
    for value in values or []:
        name, sep, adapter = value.partition("=")
        name, adapter = name.strip(), adapter.strip()
        if not sep or not name or not adapter:
            raise ValueError(f"Invalid --{option} entry {value!r}; expected NAME=ADAPTER (or NAME=none).")
        if name in result:
            raise ValueError(f"Duplicate name {name!r} in --{option}.")
        result[name] = None if adapter == "none" else adapter
    return result


def validate_opd_adapters(args: Namespace) -> None:
    """Reject incomplete routing maps before an expert silently becomes the base."""
    options = ("opd_teacher_adapters", "opd_base_adapters", "opd_anchor_adapter")
    if not any(getattr(args, option, None) is not None for option in options):
        return
    if not getattr(args, "use_opd", False) or getattr(args, "opd_type", None) != "sglang":
        raise ValueError("OPD adapter routing requires --use-opd --opd-type=sglang.")
    anchor = getattr(args, "opd_anchor_adapter", None)
    if anchor is not None and not anchor.strip():
        raise ValueError("--opd-anchor-adapter must not be empty.")
    if (getattr(args, "opd_base_adapters", None) or getattr(args, "opd_anchor_adapter", None)) and (
        _get_opd_target_mode(args) != "shiftmopd" and not paper_opd_enabled(args)
    ):
        raise ValueError("Base/anchor adapter routing requires --opd-target-mode=shiftmopd.")
    for kind in ("teacher", "base"):
        option = f"opd_{kind}_adapters"
        values = getattr(args, option, None)
        if not values:
            continue
        adapters = _adapter_map(values, option.replace("_", "-"))
        names = set(parse_teacher_urls(getattr(args, f"opd_{kind}_urls", None)))
        if kind == "teacher" and not names:
            names = {DEFAULT_TEACHER_NAME}
        if set(adapters) != names:
            raise ValueError(f"--{option.replace('_', '-')} names must match scorer routes {sorted(names)}.")


def _with_adapter(payload: dict[str, Any], adapter: str | None) -> dict[str, Any]:
    # Do not mutate the payload shared by concurrent composition requests.
    return {**payload, "lora_path": adapter} if adapter is not None else payload


def _get_opd_top_k(args: Namespace) -> int:
    return max(0, int(getattr(args, "opd_log_prob_top_k", 0) or 0))


def _get_top_k_strategy(args: Namespace) -> str:
    strategy = getattr(args, "opd_top_k_strategy", "only-student")
    if strategy not in TOP_K_STRATEGIES:
        raise ValueError(f"Unknown OPD top-k strategy: {strategy}")
    return strategy


def _get_reward_weight_mode(args: Namespace) -> str:
    mode = getattr(args, "opd_reward_weight_mode", "student_p")
    if mode not in REWARD_WEIGHT_MODES:
        raise ValueError(f"Unknown OPD reward weight mode: {mode}")
    return mode


def _get_opd_target_mode(args: Namespace) -> str:
    mode = getattr(args, "opd_target_mode", "endpoint")
    if mode not in OPD_TARGET_MODES:
        raise ValueError(f"Unknown OPD target mode: {mode}")
    return mode


def _composition_names(args: Namespace) -> list[str]:
    """Return the ordered specialist names used by ShiftMOPD composition."""
    teacher_urls = parse_teacher_urls(getattr(args, "opd_teacher_urls", None))
    if not teacher_urls:
        raise ValueError("ShiftMOPD requires --opd-teacher-urls with one entry per specialist teacher.")
    names = [name for name in teacher_urls if name != DEFAULT_TEACHER_NAME]
    if not names:
        names = [DEFAULT_TEACHER_NAME]
    return names


def _composition_support_ids(sample: Sample, student_top: TopLogprobs) -> list[list[int]]:
    """Build the paper support: student top-K IDs plus the sampled response ID."""
    prompt_len = len(sample.tokens) - sample.response_length
    sampled_ids = sample.tokens[prompt_len:]
    if len(sampled_ids) != sample.response_length:
        raise ValueError(
            f"Sampled-token length mismatch: got {len(sampled_ids)}, expected {sample.response_length}."
        )
    support_ids: list[list[int]] = []
    for entries, sampled_id in zip(student_top, sampled_ids, strict=True):
        sampled_id = int(sampled_id)
        top_ids = _ordered_unique([_top_entry_token_id(entry) for entry in (entries or []) if entry is not None])
        support_ids.append([token_id for token_id in top_ids if token_id != sampled_id] + [sampled_id])
    return support_ids


def _composition_url_map(args: Namespace, option: str, *, required: bool) -> dict[str, str]:
    values = getattr(args, option, None)
    result = parse_teacher_urls(values)
    if required and not result:
        raise ValueError(f"ShiftMOPD requires --{option.replace('_', '-')}.")
    return result


def _compute_shiftmopd_reverse_kl(anchor_log_probs, teacher_log_probs, base_log_probs,
                                student_sampled_log_probs, support_ids, **kwargs):
    """Compatibility entry point; tiled composition preserves the reward math."""
    return compose_reverse_kl(anchor_log_probs, teacher_log_probs, base_log_probs,
                              student_sampled_log_probs, support_ids, **kwargs)


def _compute_shiftmopd_reverse_kl_reference(
    anchor_log_probs: LogprobMaps,
    teacher_log_probs: list[LogprobMaps],
    base_log_probs: list[LogprobMaps],
    student_sampled_log_probs: torch.Tensor,
    support_ids: list[list[int]],
    *,
    loss_mask=None,
    center=True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute centered sampled reverse-KL and its uncentered value.

    For each response position, the supplied maps contain log-probabilities for
    the student's top-K support plus its sampled token.  Log-probability
    normalizers cancel in ``log p_A + sum(log p_T - log p_B)``, so a single
    logsumexp over the support approximates the paper's composite partition.
    The returned first tensor is the centered reverse-KL consumed by Miles'
    existing OPD advantage hook; the second is retained for diagnostics.
    """
    if len(teacher_log_probs) == 0:
        raise ValueError("ShiftMOPD requires at least one teacher.")
    if len(teacher_log_probs) != len(base_log_probs):
        raise ValueError(
            f"ShiftMOPD teacher/base count mismatch: {len(teacher_log_probs)} vs {len(base_log_probs)}."
        )
    response_length = len(support_ids)
    if len(anchor_log_probs) != response_length or student_sampled_log_probs.numel() != response_length:
        raise ValueError("ShiftMOPD scorer lengths do not match the response length.")
    if any(len(values) != response_length for values in [*teacher_log_probs, *base_log_probs]):
        raise ValueError("ShiftMOPD scorer lengths do not match the response length.")

    raw_reverse_kl = []
    for position, ids in enumerate(support_ids):
        if not ids:
            raise ValueError(f"ShiftMOPD support is empty at response position {position}.")
        if len(set(ids)) != len(ids):
            raise ValueError(f"ShiftMOPD support contains duplicate IDs at response position {position}.")
        composite_scores = []
        for token_id in ids:
            try:
                score = anchor_log_probs[position][token_id]
                for teacher, base in zip(teacher_log_probs, base_log_probs, strict=True):
                    score += teacher[position][token_id] - base[position][token_id]
            except KeyError as exc:
                raise ValueError(
                    f"ShiftMOPD scorer response is missing token {token_id} at position {position}."
                ) from exc
            composite_scores.append(score)

        composite_scores_tensor = torch.tensor(composite_scores, dtype=torch.float32)
        target_log_prob = composite_scores_tensor[-1] - torch.logsumexp(composite_scores_tensor, dim=0)
        raw_reverse_kl.append(student_sampled_log_probs[position].float() - target_log_prob)

    raw_reverse_kl_tensor = torch.stack(raw_reverse_kl)
    if not torch.isfinite(raw_reverse_kl_tensor).all():
        raise ValueError("ShiftMOPD produced a non-finite sampled reverse-KL.")
    centered_reverse_kl, _ = center_sampled_rewards(raw_reverse_kl_tensor, loss_mask, center=center)
    return centered_reverse_kl, raw_reverse_kl_tensor


def _record_opd_centering(sample, raw, centered, baseline, enabled):
    """Small per-response scalars retained in traces and rollout metrics."""
    if sample.metadata is None:
        sample.metadata = {}
    mask = torch.ones_like(raw, dtype=torch.bool) if sample.loss_mask is None else torch.as_tensor(sample.loss_mask).bool()
    n = int(mask.sum())
    sample.metadata.update(
        opd_centering_enabled=float(enabled),
        opd_centering_valid_tokens=n,
        opd_sampled_raw_reward_mean=float(raw[mask].mean()) if n else 0.,
        opd_sampled_centered_reward_mean=float(centered[mask].mean()) if n else 0.,
        opd_centering_baseline=float(baseline),
    )


def _score_payload(
    input_ids: list[int],
    top_k: int = 0,
    token_ids: list[int] | None = None,
    token_ids_positions: list[list[int]] | None = None,
) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if top_k > 0:
        payload["top_logprobs_num"] = top_k
    if token_ids_positions is not None:
        # Per-position scoring (patched sglang): one id-list per input position, so the
        # teacher returns each position's own ids (sparse) instead of the global union
        # broadcast to every position (dense O(R^2)). Aligned to logprob_start_len=0.
        payload["token_ids_logprob_positions"] = token_ids_positions
        # Prefer packed numeric buffers; retain the older raw-array request
        # for servers without the packed patch. Both preserve the same scores.
        payload["return_raw_token_ids_logprobs"] = True
        payload[REQUEST_FIELD] = True
    elif token_ids:
        payload["token_ids_logprob"] = token_ids
    return payload


def _per_position_ids(
    top_logprobs: TopLogprobs,
    prompt_len: int,
    sampled_ids: list[int] | None = None,
) -> list[list[int]]:
    """Build one token-id list per scored input position for ``token_ids_logprob_positions``.

    ``top_logprobs`` is per response position (length == response_length). Prompt
    positions are padded with empty id-lists so the layout aligns with
    ``logprob_start_len=0`` and the existing ``_trim_input_field`` extraction
    (``values[1:][-response_length:]``) — i.e. response position r lands at index
    ``prompt_len + r``.
    """
    per_pos: list[list[int]] = [[] for _ in range(prompt_len)]
    for position, entries in enumerate(top_logprobs):
        ids = [_top_entry_token_id(e) for e in (entries or []) if e is not None]
        if sampled_ids is not None:
            if len(sampled_ids) != len(top_logprobs):
                raise ValueError("Sampled-token and top-logprob lengths do not match.")
            sampled_id = int(sampled_ids[position])
            if sampled_id not in ids:
                ids.append(sampled_id)
        per_pos.append(ids)
    return per_pos


def _student_top_maps_with_sampled_token(sample: Sample, response_length: int) -> list[dict[int, float]]:
    """Return student top-k maps and explicitly retain each sampled token."""
    maps = [_top_entries_to_map(entries) for entries in _student_top_logprobs(sample, response_length)]
    if sample.rollout_log_probs is None or len(sample.rollout_log_probs) != response_length:
        raise ValueError("MOPD sampled-token support requires rollout log-probabilities.")
    sampled_ids = sample.tokens[-response_length:] if response_length > 0 else []
    for position, sampled_id in enumerate(sampled_ids):
        maps[position].setdefault(sampled_id, float(sample.rollout_log_probs[position]))
    return maps


def _student_score_url(args: Namespace) -> str:
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"


def _with_student_adapter(args: Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    """Score against the same policy adapter used by student rollout, if any."""
    if lora_rollout_enabled(args):
        return _with_adapter(payload, LORA_ADAPTER_NAME)
    return payload


async def _post_json(url: str, payload: dict[str, Any], timeout_secs: int | float | None = None) -> dict | list:
    timeout = aiohttp.ClientTimeout(total=timeout_secs)
    start = perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            body = await resp.read()
            read_done = perf_counter()
    return await asyncio.to_thread(decode_score_response, body, request_read_seconds=read_done-start)


_SCORING_LIMITERS: dict[tuple[asyncio.AbstractEventLoop, str], asyncio.Semaphore] = {}


async def _bounded_flat_score(url, payload, timeout_secs):
    # Bound simultaneous dense chunk decodes independently of student rollout
    # concurrency. Retaining K entries is not enough if hundreds of responses
    # are decoded at the same time before they can be trimmed.
    key = (asyncio.get_running_loop(), url)
    limiter = _SCORING_LIMITERS.setdefault(key, asyncio.Semaphore(8))
    async with limiter:
        return await _post_json(url, payload, timeout_secs=timeout_secs)


async def _post_json_per_position_compatible(
    url: str,
    payload: dict[str, Any],
    *,
    prompt_len: int,
    response_length: int,
    chunk_size: int = 256,
    timeout_secs: int | float | None = None,
    require_native: bool = False,
) -> dict[str, Any]:
    """Bound response memory without changing the scorer's forward context.

    Unpatched servers ignore token_ids_logprob_positions. Partition the flat
    ID union instead, preserving input_ids and logprob_start_len on every
    call. Prefix-position chunking is intentionally NOT used: cached BF16
    execution shapes can materially change composed teacher/base rewards.
    Retain only the requested entries per position, not dense chunk rows.
    Temporary response size is O(sequence_length * chunk_size), retained
    size O(response_length * K). This is a compatibility path, not native
    sparse GPU gather, and does not reduce the total dense scorer work.
    """
    response = await _post_json(url, payload, timeout_secs=timeout_secs)
    meta_info = response.get("meta_info", {})
    if any(meta_info.get(field) is not None for field in (
        "input_token_ids_logprobs", "input_token_ids_logprobs_val", "input_token_ids_logprobs_idx",
        PACKED_SCORES_FIELD, DECODED_FIELD,
    )):
        return response

    if require_native:
        raise RuntimeError(
            "Native per-position selected-ID scoring is required, but the scorer "
            "omitted input_token_ids_logprobs. Check the deployed SGLang patch."
        )

    positions = payload.get("token_ids_logprob_positions")
    if positions is None:
        return response
    if response_length <= 0:
        return response
    if prompt_len < 1:
        raise ValueError("SGLang per-position scoring requires a nonempty prompt.")
    if len(positions) < prompt_len + response_length:
        raise ValueError(
            "SGLang per-position token-id request is shorter than the prompt plus response."
        )
    if chunk_size <= 0:
        raise ValueError("SGLang per-position fallback chunk size must be positive.")

    token_maps: list[dict[int, Any]] = [{} for _ in range(response_length)]
    input_ids = payload.get("input_ids")
    if not isinstance(input_ids, list):
        raise ValueError("SGLang per-position fallback requires list input_ids.")

    wanted = [set(ids) for ids in positions[prompt_len : prompt_len + response_length]]
    union = _ordered_unique(token for row in positions[prompt_len:] for token in row)
    for start in range(0, len(union), chunk_size):
        ids = union[start : start + chunk_size]
        fallback_payload = {
            key: value
            for key, value in payload.items()
            if key != "token_ids_logprob_positions"
        }
        fallback_payload["token_ids_logprob"] = ids
        chunk_response = await _bounded_flat_score(url, fallback_payload, timeout_secs)
        chunk_meta = chunk_response.get("meta_info", {})
        chunk_tokens = chunk_meta.get("input_token_ids_logprobs")
        if chunk_tokens is None:
            raise ValueError(
                "SGLang flat token-id fallback is missing meta_info.input_token_ids_logprobs."
            )
        chunk_tokens = chunk_tokens[1:][-response_length:]
        if len(chunk_tokens) != response_length:
            raise ValueError(
                "SGLang flat token-id fallback returned an unexpected number of positions: "
                f"got {len(chunk_tokens)}, expected {response_length}."
            )
        for offset, entries in enumerate(chunk_tokens):
            token_maps[offset].update(
                (int(entry[1]), entry) for entry in entries or [] if int(entry[1]) in wanted[offset]
            )
        del chunk_response, chunk_meta, chunk_tokens

    token_rows = []
    for offset, values in enumerate(token_maps):
        missing = wanted[offset] - values.keys()
        if missing:
            raise ValueError(f"SGLang fallback is missing requested token IDs: {sorted(missing)}")
        token_rows.append([values[token] for token in _ordered_unique(positions[prompt_len + offset])])
    merged = dict(response)
    merged_meta = dict(meta_info)
    merged_meta["input_token_ids_logprobs"] = [None, *token_rows]
    merged["meta_info"] = merged_meta
    return merged


def _batch_payload(payloads: list[dict[str, Any]], adapters: list[str | None] | None = None) -> dict[str, Any]:
    """Pack compatible single-request score payloads into SGLang's batch API shape."""
    if not payloads:
        raise ValueError("Cannot build an empty SGLang score batch.")
    keys = set(payloads[0])
    if any(set(payload) != keys for payload in payloads[1:]):
        raise ValueError("SGLang score payloads in one batch must have matching fields.")

    batch = {
        "input_ids": [payload["input_ids"] for payload in payloads],
        "sampling_params": [payload["sampling_params"] for payload in payloads],
        "return_logprob": [payload["return_logprob"] for payload in payloads],
        "logprob_start_len": [payload["logprob_start_len"] for payload in payloads],
    }
    if "token_ids_logprob" in payloads[0]:
        batch["token_ids_logprob"] = [payload["token_ids_logprob"] for payload in payloads]
    if "token_ids_logprob_positions" in payloads[0]:
        batch["token_ids_logprob_positions"] = [payload["token_ids_logprob_positions"] for payload in payloads]
    for field in ("return_raw_token_ids_logprobs", REQUEST_FIELD):
        if field in payloads[0]:
            if any(payload[field] != payloads[0][field] for payload in payloads):
                raise ValueError("Scoring batches must use one transport format.")
            batch[field] = payloads[0][field]
    if adapters is not None:
        if len(adapters) != len(payloads):
            raise ValueError("SGLang batch adapter count must match payload count.")
        if any(adapter is not None for adapter in adapters):
            batch["lora_path"] = adapters
    return batch


async def _post_json_batch(
    url: str,
    payloads: list[dict[str, Any]],
    adapters: list[str | None] | None = None,
    timeout_secs: int | float | None = None,
) -> list[dict[str, Any]]:
    """Score multiple sequences in one SGLang ``/generate`` request."""
    result = await _post_json(url, _batch_payload(payloads, adapters), timeout_secs=timeout_secs)
    if isinstance(result, dict) and isinstance(result.get("responses"), list):
        result = result["responses"]
    if not isinstance(result, list) or len(result) != len(payloads):
        raise ValueError(
            f"SGLang batch response must contain {len(payloads)} responses, got {type(result).__name__}"
        )
    return result


async def _score_unique_routes(
    routes: list[tuple[str, str | None]],
    payload: dict[str, Any],
    timeout_secs: int | float | None,
    fallback_shape: tuple[int, int, int] | None = None,
    require_native: bool = False,
) -> dict[tuple[str, str | None], dict[str, Any]]:
    # A shared URL is not a shared model when different adapters are selected.
    # Only identical (endpoint, adapter) pairs can reuse this sample's scores.
    unique_routes = list(dict.fromkeys(routes))
    if fallback_shape is None:
        responses = await asyncio.gather(
            *(_post_json(url, _with_adapter(payload, adapter), timeout_secs) for url, adapter in unique_routes)
        )
    else:
        prompt_len, response_length, chunk_size = fallback_shape
        responses = await asyncio.gather(
            *(
                _post_json_per_position_compatible(
                    url,
                    _with_adapter(payload, adapter),
                    prompt_len=prompt_len,
                    response_length=response_length,
                    chunk_size=chunk_size,
                    timeout_secs=timeout_secs,
                    require_native=require_native,
                )
                for url, adapter in unique_routes
            )
        )
    return dict(zip(unique_routes, responses, strict=True))


def _top_entry_token_id(entry: list[Any]) -> int:
    return int(entry[1])


def _top_entry_logprob(entry: list[Any]) -> float:
    return float(entry[0])


def _top_entries_to_map(entries: Iterable[list[Any]] | None) -> dict[int, float]:
    if not entries:
        return {}
    return {_top_entry_token_id(entry): _top_entry_logprob(entry) for entry in entries if entry is not None}


def _trim_input_field(meta_info: dict[str, Any], field: str, response_length: int) -> list[Any]:
    if field == "input_token_ids_logprobs":
        return selected_logprob_rows(meta_info, response_length)
    values = meta_info.get(field)
    if values is None:
        raise ValueError(f"Teacher response is missing meta_info.{field}.")
    # SGLang's first input logprob/top-logprob position is a placeholder.
    return values[1:][-response_length:] if response_length > 0 else []


def _input_logprob_maps(response: dict[str, Any], field: str, response_length: int) -> LogprobMaps:
    return [
        _top_entries_to_map(entries) for entries in _trim_input_field(response["meta_info"], field, response_length)
    ]


def _teacher_sampled_log_probs(response: dict[str, Any], response_length: int) -> torch.Tensor:
    input_token_logprobs = _trim_input_field(response["meta_info"], "input_token_logprobs", response_length)
    return torch.tensor([item[0] for item in input_token_logprobs], dtype=torch.float32)


def _student_top_logprobs(sample: Sample, response_length: int) -> TopLogprobs:
    top_logprobs = sample.metadata.get("opd_student_top_logprobs")
    if top_logprobs is None:
        raise ValueError(
            "Top-k OPD requires student output_top_logprobs. "
            "Ensure --opd-log-prob-top-k is set before rollout generation starts."
        )
    top_logprobs = top_logprobs[-response_length:] if response_length > 0 else []
    if len(top_logprobs) != response_length:
        raise ValueError(
            f"Student top-k logprob length mismatch: got {len(top_logprobs)}, expected {response_length}."
        )
    return top_logprobs


def _endpoint_batch_request(
    args: Namespace, sample: Sample
) -> tuple[tuple[str, str | None], dict[str, Any]] | None:
    """Build the one-request endpoint OPD payload used by the batched path.

    Strategies that require a second student-on-teacher request are left on the
    existing per-sample path for now. The full-run experiments use
    ``only-student``, which can be batched directly.
    """
    strategy = _get_top_k_strategy(args)
    if strategy in STUDENT_ON_TEACHER_STRATEGIES:
        return None
    teacher_name, teacher_url = _teacher_route_for_sample(args, sample)
    teacher_adapters = _adapter_map(getattr(args, "opd_teacher_adapters", None), "opd-teacher-adapters")
    teacher_adapter = teacher_adapters.get(teacher_name)
    top_k = _get_opd_top_k(args)
    if top_k == 0:
        payload = _score_payload(sample.tokens)
    else:
        student_top = _student_top_logprobs(sample, sample.response_length)
        if getattr(args, "opd_topk_per_position", False):
            payload = _score_payload(
                sample.tokens,
                token_ids_positions=_per_position_ids(
                    student_top,
                    len(sample.tokens) - sample.response_length,
                    sample.tokens[-sample.response_length :] if sample.response_length > 0 else [],
                ),
            )
        else:
            payload = _score_payload(
                sample.tokens,
                token_ids=_ordered_unique(
                    [
                        *_unique_ids(student_top),
                        *(sample.tokens[-sample.response_length :] if sample.response_length > 0 else []),
                    ]
                ),
            )
    return (teacher_url, teacher_adapter), payload


def _unique_ids(top_logprobs: Iterable[Iterable[list[Any]]]) -> list[int]:
    ids = set()
    for entries in top_logprobs:
        for entry in entries or []:
            if entry is not None:
                ids.add(_top_entry_token_id(entry))
    return sorted(ids)


def _ordered_unique(ids: Iterable[int]) -> list[int]:
    seen = set()
    ordered = []
    for token_id in ids:
        if token_id in seen:
            continue
        seen.add(token_id)
        ordered.append(token_id)
    return ordered


def _selected_token_ids(strategy: str, student_ids: list[int], teacher_ids: list[int]) -> list[int]:
    student_set = set(student_ids)
    teacher_set = set(teacher_ids)
    if strategy == "only-student":
        return student_ids
    if strategy == "only-teacher":
        return teacher_ids
    if strategy == "intersection":
        return [token_id for token_id in student_ids if token_id in teacher_set]
    if strategy == "union":
        return _ordered_unique([*student_ids, *teacher_ids])
    if strategy == "xor":
        return [
            token_id
            for token_id in [*student_ids, *teacher_ids]
            if (token_id in student_set) != (token_id in teacher_set)
        ]
    raise ValueError(f"Unknown OPD top-k strategy: {strategy}")


def _lookup_logprob(
    token_id: int,
    primary: dict[int, float],
    fallback: dict[int, float] | None,
    *,
    source: str,
) -> float:
    if token_id in primary:
        return primary[token_id]
    if fallback is not None and token_id in fallback:
        return fallback[token_id]
    raise ValueError(f"Missing {source} logprob for token id {token_id}.")


def _reward_weights(
    student_logps: list[float],
    teacher_logps: list[float],
    mode: str,
    *,
    normalize: bool,
) -> list[float]:
    if not student_logps:
        return []
    if mode == "student_p":
        logps = student_logps
    elif mode == "teacher_p":
        logps = teacher_logps
    elif mode == "none":
        logps = [0.0] * len(student_logps)
    else:
        raise ValueError(f"Unknown OPD reward weight mode: {mode}")

    if not normalize:
        return [math.exp(logp) for logp in logps]

    max_logp = max(logps)
    exp_vals = [math.exp(logp - max_logp) for logp in logps]
    denom = sum(exp_vals)
    if denom == 0.0:
        return [0.0] * len(logps)
    return [v / denom for v in exp_vals]


def _candidate_sampled_baseline(sample: Sample, teacher_response, teacher_on_student_maps) -> float:
    """Mean over sampled positions, never over the candidate support."""
    n = sample.response_length
    if sample.rollout_log_probs is None or len(sample.rollout_log_probs) != n:
        raise ValueError("Centered candidate OPD requires aligned sampled rollout log-probabilities.")
    sampled_ids = sample.tokens[-n:]
    # Selected-ID results contain sampled IDs for only-student. Other strategies
    # may omit them, so fall back to the ordinary prefill log-probabilities.
    if all(t in row for t, row in zip(sampled_ids, teacher_on_student_maps, strict=True)):
        sampled_teacher = torch.tensor([row[t] for t, row in zip(sampled_ids, teacher_on_student_maps, strict=True)])
    else:
        sampled_teacher = _teacher_sampled_log_probs(teacher_response, n)
    sampled_student = torch.tensor(sample.rollout_log_probs, dtype=torch.float32)
    if sampled_teacher.shape != sampled_student.shape:
        raise ValueError("Centered candidate OPD teacher/student response length mismatch.")
    raw_sampled = sampled_teacher - sampled_student
    centered_sampled, baseline = center_sampled_rewards(raw_sampled, sample.loss_mask)
    _record_opd_centering(sample, raw_sampled, centered_sampled, baseline, True)
    return float(baseline)


def _compute_topk_candidates(
    args: Namespace,
    sample: Sample,
    reward_payload: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Keep candidate IDs, old log-probs and weighted rewards until PPO clipping.

    Weight normalization is over the selected support, but log-ratios retain
    full-vocabulary log-probabilities. Optional centering subtracts the valid
    sampled-token response baseline BEFORE candidate weighting. No whitening.
    """
    response_length = sample.response_length
    if response_length == 0:
        return {
            "opd_candidate_ids": torch.zeros((0, 1), dtype=torch.long),
            "opd_candidate_old_log_probs": torch.zeros((0, 1)),
            "opd_candidate_rewards": torch.zeros((0, 1)),
            "opd_candidate_mask": torch.zeros((0, 1), dtype=torch.bool),
        }

    strategy = _get_top_k_strategy(args)
    weight_mode = _get_reward_weight_mode(args)
    if strategy in STUDENT_ON_TEACHER_STRATEGIES and getattr(args, "rollout_temperature", 1.0) != 1.0:
        raise ValueError(
            "Teacher-selected candidate supports require rollout_temperature=1: "
            "student input rescoring returns untempered log-probabilities. "
            "Use only-student for the temperature-scaled top-K baseline."
        )

    student_top_maps = (
        _student_top_maps_with_sampled_token(sample, response_length)
        if strategy in STUDENT_TOP_STRATEGIES
        else [{} for _ in range(response_length)]
    )

    teacher_response = reward_payload["teacher"]
    teacher_top_maps = (
        _input_logprob_maps(teacher_response, "input_top_logprobs", response_length)
        if strategy in TEACHER_TOP_STRATEGIES
        else [{} for _ in range(response_length)]
    )
    teacher_on_student_maps = (
        _input_logprob_maps(teacher_response, "input_token_ids_logprobs", response_length)
        if strategy in TEACHER_ON_STUDENT_STRATEGIES
        else [{} for _ in range(response_length)]
    )
    student_on_teacher_maps = (
        _input_logprob_maps(reward_payload["student_on_teacher"], "input_token_ids_logprobs", response_length)
        if strategy in STUDENT_ON_TEACHER_STRATEGIES
        else [{} for _ in range(response_length)]
    )

    baseline = 0.0
    if opd_centering_mode(args) == "response":
        baseline = _candidate_sampled_baseline(sample, teacher_response, teacher_on_student_maps)

    rows = []
    normalize_weights = strategy != "xor"
    for i in range(response_length):
        student_ids = list(student_top_maps[i].keys())
        teacher_ids = list(teacher_top_maps[i].keys())
        selected_ids = _selected_token_ids(strategy, student_ids, teacher_ids)

        student_logps = []
        teacher_logps = []
        for token_id in selected_ids:
            student_logps.append(
                _lookup_logprob(
                    token_id,
                    student_top_maps[i],
                    student_on_teacher_maps[i],
                    source="student",
                )
            )
            teacher_logps.append(
                _lookup_logprob(
                    token_id,
                    teacher_top_maps[i],
                    teacher_on_student_maps[i],
                    source="teacher",
                )
            )

        weights = _reward_weights(student_logps, teacher_logps, weight_mode, normalize=normalize_weights)
        rewards = [
            w * (t_logp - s_logp - baseline) for w, s_logp, t_logp in zip(weights, student_logps, teacher_logps, strict=True)
        ]
        if not all(math.isfinite(value) for value in [*student_logps, *teacher_logps, *rewards]):
            raise ValueError("Non-finite OPD candidate score.")
        rows.append((selected_ids, student_logps, rewards))

    width = max((len(ids) for ids, _, _ in rows), default=0)
    shape = (response_length, max(1, width))
    result = {
        "opd_candidate_ids": torch.zeros(shape, dtype=torch.long),
        "opd_candidate_old_log_probs": torch.zeros(shape, dtype=torch.float32),
        "opd_candidate_rewards": torch.zeros(shape, dtype=torch.float32),
        "opd_candidate_mask": torch.zeros(shape, dtype=torch.bool),
    }
    for i, (ids, old_logps, rewards) in enumerate(rows):
        n = len(ids)
        result["opd_candidate_ids"][i, :n] = torch.tensor(ids, dtype=torch.long)
        result["opd_candidate_old_log_probs"][i, :n] = torch.tensor(old_logps)
        result["opd_candidate_rewards"][i, :n] = torch.tensor(rewards)
        result["opd_candidate_mask"][i, :n] = True
    return result


def _compute_topk_reverse_kl(args: Namespace, sample: Sample, reward_payload: dict[str, Any]) -> torch.Tensor:
    """Scalar diagnostic only; never use this reduction as a sampled-action advantage."""
    return -_compute_topk_candidates(args, sample, reward_payload)["opd_candidate_rewards"].sum(dim=-1)


async def _reward_func_single(args: Namespace, sample: Sample, **kwargs: Any) -> dict[str, Any]:
    validate_opd_adapters(args)
    teacher_adapters = _adapter_map(getattr(args, "opd_teacher_adapters", None), "opd-teacher-adapters")
    top_k = _get_opd_top_k(args)
    # Optional per-request timeout so a hung teacher/student scoring call cannot stall
    # the whole rollout (no-op when unset).
    request_timeout = getattr(args, "sglang_router_request_timeout_secs", None)
    target_mode = _get_opd_target_mode(args)

    if target_mode == "shiftmopd":
        if top_k <= 0:
            raise ValueError("ShiftMOPD requires --opd-log-prob-top-k > 0.")
        student_top = _student_top_logprobs(sample, sample.response_length)
        support_ids = _composition_support_ids(sample, student_top)
        teacher_urls = parse_teacher_urls(getattr(args, "opd_teacher_urls", None))
        base_urls = _composition_url_map(args, "opd_base_urls", required=True)
        anchor_url = getattr(args, "opd_anchor_url", None)
        if not anchor_url:
            raise ValueError("ShiftMOPD requires --opd-anchor-url.")
        names = _composition_names(args)
        if set(names) != set(base_urls):
            raise ValueError(
                f"ShiftMOPD teacher/base names must match: teachers={sorted(names)}, bases={sorted(base_urls)}."
            )

        if getattr(args, "opd_topk_per_position", False):
            payload = _score_payload(
                sample.tokens,
                token_ids_positions=[[] for _ in range(len(sample.tokens) - sample.response_length)]
                + support_ids,
            )
        else:
            # The released SGLang server accepts a flat token_ids_logprob list,
            # applying that selected student-token union at every position.
            # Keep the sampled response IDs in the support even when they are
            # outside the student's top-k union.
            global_ids = _ordered_unique(
                [*_unique_ids(student_top), *sample.tokens[-sample.response_length :]]
            )
            payload = _score_payload(sample.tokens, token_ids=global_ids)
        base_adapters = _adapter_map(getattr(args, "opd_base_adapters", None), "opd-base-adapters")
        anchor_adapter = getattr(args, "opd_anchor_adapter", None)
        if anchor_adapter is not None:
            anchor_adapter = anchor_adapter.strip()
        anchor = (anchor_url, None if anchor_adapter == "none" else anchor_adapter)
        teachers = {name: (teacher_urls[name], teacher_adapters.get(name)) for name in names}
        bases = {name: (base_urls[name], base_adapters.get(name)) for name in names}
        responses = await _score_unique_routes(
            [anchor, *teachers.values(), *bases.values()],
            payload,
            request_timeout,
            (
                len(sample.tokens) - sample.response_length,
                sample.response_length,
                max(1, int(getattr(args, "opd_topk_fallback_chunk_size", 256) or 256)),
            )
            if getattr(args, "opd_topk_per_position", False)
            else None,
            require_native=getattr(args, "opd_require_native_selected_ids", False),
        )
        return {
            "mode": "shiftmopd",
            "support_ids": support_ids,
            "anchor": responses[anchor],
            "teachers": {name: responses[route] for name, route in teachers.items()},
            "bases": {name: responses[route] for name, route in bases.items()},
        }

    # Multi-teacher routing: pick this sample's teacher endpoint (falls back to
    # --rm-url when --opd-teacher-urls is unset).
    teacher_name, teacher_url = _teacher_route_for_sample(args, sample)
    teacher_adapter = teacher_adapters.get(teacher_name)
    if top_k == 0:
        return await _post_json(
            teacher_url, _with_adapter(_score_payload(sample.tokens), teacher_adapter), timeout_secs=request_timeout
        )

    strategy = _get_top_k_strategy(args)
    # Per-position scoring requires a patched teacher/student server that understands
    # token_ids_logprob_positions; default off so an unpatched server keeps working.
    per_position = getattr(args, "opd_topk_per_position", False)
    prompt_len = len(sample.tokens) - sample.response_length

    teacher_top_k = top_k if strategy in TEACHER_TOP_STRATEGIES else 0
    if strategy in TEACHER_ON_STUDENT_STRATEGIES:
        student_top = _student_top_logprobs(sample, sample.response_length)
        teacher_token_ids = _ordered_unique(
            [
                *_unique_ids(student_top),
                *(sample.tokens[-sample.response_length :] if sample.response_length > 0 else []),
            ]
        )
    else:
        student_top = None
        teacher_token_ids = None

    if student_top is not None and per_position:
        teacher_payload = _score_payload(
            sample.tokens,
            top_k=teacher_top_k,
            token_ids_positions=_per_position_ids(
                student_top,
                prompt_len,
                sample.tokens[-sample.response_length :] if sample.response_length > 0 else [],
            ),
        )
    elif teacher_token_ids is not None:
        teacher_payload = _score_payload(sample.tokens, top_k=teacher_top_k, token_ids=teacher_token_ids)
    else:
        teacher_payload = _score_payload(sample.tokens, top_k=teacher_top_k)
    if per_position:
        teacher_response = await _post_json_per_position_compatible(
            teacher_url,
            _with_adapter(teacher_payload, teacher_adapter),
            prompt_len=prompt_len,
            response_length=sample.response_length,
            chunk_size=max(1, int(getattr(args, "opd_topk_fallback_chunk_size", 256) or 256)),
            timeout_secs=request_timeout,
            require_native=getattr(args, "opd_require_native_selected_ids", False),
        )
    else:
        teacher_response = await _post_json(
            teacher_url, _with_adapter(teacher_payload, teacher_adapter), timeout_secs=request_timeout
        )

    reward_payload = {"teacher": teacher_response}
    if strategy in STUDENT_ON_TEACHER_STRATEGIES:
        teacher_top = _trim_input_field(teacher_response["meta_info"], "input_top_logprobs", sample.response_length)
        if per_position:
            student_payload = _score_payload(
                sample.tokens, token_ids_positions=_per_position_ids(teacher_top, prompt_len)
            )
        else:
            student_payload = _score_payload(sample.tokens, token_ids=_unique_ids(teacher_top))
        if per_position:
            reward_payload["student_on_teacher"] = await _post_json_per_position_compatible(
                _student_score_url(args),
                _with_student_adapter(args, student_payload),
                prompt_len=prompt_len,
                response_length=sample.response_length,
                chunk_size=max(1, int(getattr(args, "opd_topk_fallback_chunk_size", 256) or 256)),
                timeout_secs=request_timeout,
                require_native=getattr(args, "opd_require_native_selected_ids", False),
            )
        else:
            reward_payload["student_on_teacher"] = await _post_json(
                _student_score_url(args),
                _with_student_adapter(args, student_payload),
                timeout_secs=request_timeout,
            )

    return reward_payload


async def _score_batched_routes(
    route_payloads: dict[tuple[str, str | None], list[dict[str, Any]]],
    timeout_secs: int | float | None,
) -> dict[tuple[str, str | None], list[dict[str, Any]]]:
    """Issue one native SGLang batch request for every distinct route."""
    routes = list(route_payloads)
    responses = await asyncio.gather(
        *(
            _post_json_batch(
                url,
                route_payloads[(url, adapter)],
                adapters=[adapter] * len(route_payloads[(url, adapter)]),
                timeout_secs=timeout_secs,
            )
            for url, adapter in routes
        )
    )
    return dict(zip(routes, responses, strict=True))


async def _reward_func_batch(args: Namespace, samples: list[Sample], **kwargs: Any) -> list[dict[str, Any]]:
    """Compute OPD score payloads for many generated samples at once."""
    if not samples:
        return []
    del kwargs
    validate_opd_adapters(args)
    request_timeout = getattr(args, "sglang_router_request_timeout_secs", None)
    # The released SGLang batch API has no per-position token-id field.  Use
    # the compatibility-aware single-sample path, which chunks only when the
    # server ignores token_ids_logprob_positions.  This remains concurrent
    # across samples and avoids constructing a dense batch fallback.
    if getattr(args, "opd_topk_per_position", False):
        return await asyncio.gather(*(_reward_func_single(args, sample) for sample in samples))
    target_mode = _get_opd_target_mode(args)

    if target_mode == "shiftmopd":
        teacher_adapters = _adapter_map(getattr(args, "opd_teacher_adapters", None), "opd-teacher-adapters")
        base_adapters = _adapter_map(getattr(args, "opd_base_adapters", None), "opd-base-adapters")
        anchor_adapter = getattr(args, "opd_anchor_adapter", None)
        anchor_adapter = None if anchor_adapter in (None, "none") else anchor_adapter.strip()
        teacher_urls = parse_teacher_urls(getattr(args, "opd_teacher_urls", None))
        base_urls = _composition_url_map(args, "opd_base_urls", required=True)
        names = _composition_names(args)
        anchor_url = getattr(args, "opd_anchor_url", None)
        if not anchor_url:
            raise ValueError("ShiftMOPD requires --opd-anchor-url.")

        supports: list[list[list[int]]] = []
        payloads: list[dict[str, Any]] = []
        for sample in samples:
            student_top = _student_top_logprobs(sample, sample.response_length)
            supports.append(_composition_support_ids(sample, student_top))
            if getattr(args, "opd_topk_per_position", False):
                payloads.append(
                    _score_payload(
                        sample.tokens,
                        token_ids_positions=_per_position_ids(
                            student_top, len(sample.tokens) - sample.response_length
                        ),
                    )
                )
            else:
                global_ids = _ordered_unique(
                    [*_unique_ids(student_top), *sample.tokens[-sample.response_length :]]
                )
                payloads.append(_score_payload(sample.tokens, token_ids=global_ids))

        anchor = (anchor_url, anchor_adapter)
        teachers = {name: (teacher_urls[name], teacher_adapters.get(name)) for name in names}
        bases = {name: (base_urls[name], base_adapters.get(name)) for name in names}
        route_payloads: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
        for route in [anchor, *teachers.values(), *bases.values()]:
            route_payloads.setdefault(route, payloads)
        route_responses = await _score_batched_routes(route_payloads, request_timeout)
        return [
            {
                "mode": "shiftmopd",
                "support_ids": supports[index],
                "anchor": route_responses[anchor][index],
                "teachers": {name: route_responses[route][index] for name, route in teachers.items()},
                "bases": {name: route_responses[route][index] for name, route in bases.items()},
            }
            for index in range(len(samples))
        ]

    # The full MOPD run uses only-student top-k (or sampled-token OPD), both of
    # which need one teacher request per sample. More elaborate strategies that
    # require a second student-on-teacher request stay on the proven single path.
    requests: list[tuple[tuple[str, str | None], dict[str, Any]] | None] = [
        _endpoint_batch_request(args, sample) for sample in samples
    ]
    if any(request is None for request in requests):
        return await asyncio.gather(*(_reward_func_single(args, sample) for sample in samples))

    route_payloads: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    sample_routes: list[tuple[str, str | None]] = []
    for request in requests:
        assert request is not None
        route, payload = request
        sample_routes.append(route)
        route_payloads.setdefault(route, []).append(payload)
    route_responses = await _score_batched_routes(route_payloads, request_timeout)
    route_offsets = {route: 0 for route in route_payloads}
    rewards = []
    for route in sample_routes:
        offset = route_offsets[route]
        rewards.append({"teacher": route_responses[route][offset]})
        route_offsets[route] += 1
    return rewards


class _OpdRewardBatcher:
    """Coalesce per-sample RM calls while rollout generation is still concurrent."""

    def __init__(self, args: Namespace):
        self.args = args
        self.pending: list[tuple[Sample, asyncio.Future]] = []
        self.flush_task: asyncio.Task | None = None

    async def submit(self, sample: Sample) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.pending.append((sample, future))
        batch_size = max(1, int(getattr(self.args, "opd_reward_batch_size", 0) or 1))
        if len(self.pending) >= batch_size:
            self._schedule_flush(0.0)
        elif self.flush_task is None:
            wait_ms = max(0.0, float(getattr(self.args, "opd_reward_batch_wait_ms", 25.0)))
            self._schedule_flush(wait_ms / 1000.0)
        return await future

    def _schedule_flush(self, delay: float) -> None:
        if self.flush_task is None:
            self.flush_task = asyncio.create_task(self._flush_after(delay))

    async def _flush_after(self, delay: float) -> None:
        try:
            if delay:
                await asyncio.sleep(delay)
            self.flush_task = None
            await self._flush()
        except asyncio.CancelledError:
            self.flush_task = None
            raise

    async def _flush(self) -> None:
        if not self.pending:
            return
        batch_size = max(1, int(getattr(self.args, "opd_reward_batch_size", 0) or 1))
        items = self.pending[:batch_size]
        del self.pending[:batch_size]
        samples = [sample for sample, _ in items]
        try:
            rewards = await _reward_func_batch(self.args, samples)
        except BaseException as exc:
            for _, future in items:
                if not future.done():
                    future.set_exception(exc)
        else:
            for (_, future), reward in zip(items, rewards, strict=True):
                if not future.done():
                    future.set_result(reward)
        if self.pending:
            self._schedule_flush(0.0 if len(self.pending) >= batch_size else 0.025)


_OPD_REWARD_BATCHERS: dict[tuple[int, int], _OpdRewardBatcher] = {}


def _get_opd_reward_batcher(args: Namespace) -> _OpdRewardBatcher:
    loop = asyncio.get_running_loop()
    key = (id(loop), id(args))
    batcher = _OPD_REWARD_BATCHERS.get(key)
    if batcher is None:
        batcher = _OpdRewardBatcher(args)
        _OPD_REWARD_BATCHERS[key] = batcher
    return batcher


async def reward_func(args: Namespace, sample: Sample | list[Sample], **kwargs: Any):
    """OPD reward hook with optional SGLang request batching."""
    if paper_opd_enabled(args):
        # Deferred to avoid a cycle: the paper adapter reuses the wire helpers.
        from miles.rollout.paper_opd import reward_func as paper_reward_func

        return await paper_reward_func(args, sample, **kwargs)
    if isinstance(sample, list):
        return await _reward_func_batch(args, sample, **kwargs)
    if int(getattr(args, "opd_reward_batch_size", 0) or 0) > 1:
        return await _get_opd_reward_batcher(args).submit(sample)
    return await _reward_func_single(args, sample, **kwargs)


def post_process_rewards(args: Namespace, samples: list[Sample], **kwargs: Any) -> tuple[list[float], list[float]]:
    """Extract OPD signals from teacher responses.

    ``--opd-log-prob-top-k=0`` preserves the original sampled-token OPD path:
    store teacher log-probs and let training compute ``student_logp - teacher_logp``.

    ``--opd-log-prob-top-k>0`` follows the practical recipe from
    "Rethinking On-Policy Distillation" by forming a top-k token set per
    response position and storing unreduced candidate rewards for candidate-wise
    PPO. The scalar reverse-KL reduction is retained only as a diagnostic.
    """
    if paper_opd_enabled(args):
        # Deferred for the same transport-adapter dependency as reward_func.
        from miles.rollout.paper_opd import post_process_rewards as paper_post_process

        return paper_post_process(args, samples, **kwargs)
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    if _get_opd_target_mode(args) == "shiftmopd":
        for sample, reward in zip(samples, raw_rewards, strict=True):
            if not isinstance(reward, dict) or reward.get("mode") != "shiftmopd":
                raise ValueError("ShiftMOPD post-processing expected a structured ShiftMOPD scorer response.")
            if sample.response_length == 0:
                sample.opd_reverse_kl = []
                continue
            if sample.rollout_log_probs is None:
                raise ValueError("ShiftMOPD requires student rollout log-probabilities.")
            backend = getattr(args, "opd_composition_backend", "vectorized")
            if backend not in ("vectorized", "reference"):
                raise ValueError(f"Unknown OPD composition backend: {backend}")
            centered, raw, timings = compose_scorer_responses(
                reward,
                torch.tensor(sample.rollout_log_probs, dtype=torch.float32),
                loss_mask=sample.loss_mask,
                center=opd_centering_mode(args) == "response",
                tile_size=getattr(args, "opd_composition_tile_size", 2048),
                reference=_compute_shiftmopd_reverse_kl_reference if backend == "reference" else None,
            )
            centered_reward, baseline = center_sampled_rewards(
                -raw, sample.loss_mask, center=opd_centering_mode(args) == "response"
            )
            _record_opd_centering(sample, -raw, centered_reward, baseline, opd_centering_mode(args) == "response")
            sample.opd_reverse_kl = centered.tolist()
            if sample.metadata is None:
                sample.metadata = {}
            sample.metadata["opd_perf"] = timings
            sample.metadata["shiftmopd_raw_reverse_kl_mean"] = float(raw.mean().item())
            sample.metadata["shiftmopd_raw_reward_mean"] = float((-raw).mean().item())
            sample.metadata["shiftmopd_centered_reward_mean"] = float((-centered).mean().item())

        # The OPD hook subtracts opd_reverse_kl from the base advantage. With
        # zero task reward this is exactly the paper's centered sampled reward.
        scalar_rewards = [0.0] * len(samples)
        return scalar_rewards, scalar_rewards

    if _get_opd_top_k(args) > 0:
        for sample, reward in zip(samples, raw_rewards, strict=True):
            candidates = _compute_topk_candidates(args, sample, reward)
            if sample.metadata is None:
                sample.metadata = {}
            sample.metadata["opd_candidates"] = candidates
            transports = {id(value): value for value in reward.values() if isinstance(value, dict)}
            metrics = {}
            for response in transports.values():
                for key, value in response.get("_opd_transport", {}).items():
                    metrics[key] = metrics.get(key, 0.) + value
            sample.metadata["opd_perf"] = metrics
            sample.opd_reverse_kl = -candidates["opd_candidate_rewards"].sum(dim=-1)
        scalar_rewards = [0.0] * len(samples)
        return scalar_rewards, scalar_rewards

    teacher_log_probs = [
        _teacher_sampled_log_probs(reward, response_length)
        for reward, response_length in zip(raw_rewards, response_lengths, strict=True)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=True):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator.
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
