"""Adapt SGLang selected-input-logprob wire formats to response-aligned rows."""

import json
from time import perf_counter

from miles.utils.selected_logprob_transport import (
    DECODED_FIELD,
    FIELD,
    SelectedLogprobBuffer,
    decode_selected_logprobs,
    unpack_selected_logprobs,
)


def decode_score_response(body: bytes, request_read_seconds: float = 0.) -> dict | list:
    """HTTP worker-thread decoder shared by single and batched scoring.

    Discard base64 before downstream/Ray transport. Batch-wide byte/read/JSON
    costs are divided among items so summing telemetry counts each cost once.
    Packed decoding time is measured separately for each response.
    """
    start = perf_counter()
    result = json.loads(body)
    json_seconds = perf_counter() - start
    if not isinstance(result, (dict, list)):
        raise ValueError("Scorer response must be an object or a batch of objects.")
    responses = result.get("responses", [result]) if isinstance(result, dict) else result
    if not isinstance(responses, list) or any(not isinstance(item, dict) for item in responses):
        raise ValueError("Scorer batch must contain response objects.")
    for item in responses:
        meta = item.get("meta_info", {})
        if not isinstance(meta, dict):
            raise ValueError("Scorer metadata must be an object.")
        # Never trust a server-supplied value as a locally validated buffer.
        meta.pop(DECODED_FIELD, None)
        packed = FIELD in meta
        start = perf_counter()
        if packed:
            meta[DECODED_FIELD] = decode_selected_logprobs(meta.pop(FIELD))
        item["_opd_transport"] = dict(
            response_bytes=len(body) / len(responses),
            request_read_seconds=request_read_seconds / len(responses),
            decode_seconds=json_seconds / len(responses),
            packed_decode_seconds=perf_counter() - start if packed else 0.,
            packed_response=float(packed),
        )
    return result


def selected_logprob_rows(meta_info: dict, response_length: int) -> list:
    """Return response-aligned (logprob, token ID, optional text) rows.

    Nested triples include SGLang's leading placeholder. Raw/packed arrays
    do not; all may include prompt rows. Packed rows remain read-only views.
    """
    if response_length < 0:
        raise ValueError("Response length must be nonnegative.")
    if DECODED_FIELD in meta_info:
        buffer = meta_info[DECODED_FIELD]
        if not isinstance(buffer, SelectedLogprobBuffer):
            raise ValueError("Selected score buffer must be locally decoded.")
        return buffer.response_rows(response_length)
    if FIELD in meta_info:
        return unpack_selected_logprobs(meta_info[FIELD], response_length)
    values = meta_info.get("input_token_ids_logprobs_val")
    ids = meta_info.get("input_token_ids_logprobs_idx")
    if values is not None or ids is not None:
        if values is None or ids is None or len(values) != len(ids):
            raise ValueError("Raw scorer value/id arrays are incomplete or misaligned.")
        if len(values) < response_length:
            raise ValueError("Selected logprob rows are shorter than the response.")
        if response_length == 0:
            return []
        return [list(zip(row_values or [], row_ids or [], strict=True))
                for row_values, row_ids in zip(values[-response_length:], ids[-response_length:], strict=True)]

    rows = meta_info.get("input_token_ids_logprobs")
    if rows is None:
        raise ValueError("Teacher response is missing selected-token logprob fields.")
    rows = rows[1:]
    if len(rows) < response_length:
        raise ValueError("Selected logprob rows are shorter than the response.")
    return rows[-response_length:] if response_length else []
