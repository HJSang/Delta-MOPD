"""Lossless ragged selected-score wire format; also installed into SGLang.

Only the HTTP representation changes. Preserve IDs/order and use FP32 only
when every FP64 input is exactly representable; otherwise retain FP64.
This module has no Miles, Torch, or SGLang dependencies.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pybase64


FIELD = "input_token_ids_logprobs_packed"
REQUEST_FIELD = "return_packed_token_ids_logprobs"
DECODED_FIELD = "_selected_logprob_buffer"


@dataclass(frozen=True)
class SelectedLogprobRow:
    values: np.ndarray
    token_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[tuple[float, int]]:
        return zip(map(float, self.values), map(int, self.token_ids), strict=True)


@dataclass(frozen=True)
class SelectedLogprobBuffer:
    """Locally validated flat buffers; keep compact through process transport."""

    offsets: np.ndarray
    values: np.ndarray
    token_ids: np.ndarray

    def response_rows(self, response_length: int) -> list[SelectedLogprobRow]:
        rows = len(self.offsets) - 1
        if response_length < 0 or response_length > rows:
            raise ValueError("Packed selected scores are shorter than the response.")
        return [SelectedLogprobRow(self.values[self.offsets[i]:self.offsets[i+1]],
                                  self.token_ids[self.offsets[i]:self.offsets[i+1]])
                for i in range(rows - response_length, rows)]


def _encode(array: np.ndarray) -> str:
    return pybase64.b64encode(array.tobytes()).decode("ascii")


def pack_selected_logprobs(values: list, token_ids: list) -> dict:
    """Pack scheduler rows without altering row boundaries, IDs, or score bits."""
    if len(values) != len(token_ids):
        raise ValueError("Selected score value/ID row counts differ.")
    lengths = []
    for row_values, row_ids in zip(values, token_ids, strict=True):
        if (row_values is None) != (row_ids is None):
            raise ValueError("Selected score null rows differ.")
        n = 0 if row_values is None else len(row_values)
        if n != (0 if row_ids is None else len(row_ids)):
            raise ValueError("Selected score value/ID row widths differ.")
        lengths.append(n)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64))).astype("<i8")
    count = int(offsets[-1])
    scores = np.fromiter((v for row in values if row is not None for v in row), dtype="<f8", count=count)
    # Validate integer identity before narrowing; never silently wrap token IDs.
    ids = np.fromiter((i for row in token_ids if row is not None for i in row), dtype="<i8", count=count)
    if any(type(i) is not int and not isinstance(i, np.integer)
           for row in token_ids if row is not None for i in row):
        raise ValueError("Selected token IDs must be integers.")
    if np.any(ids < 0) or np.any(ids > np.iinfo(np.int32).max):
        raise ValueError("Selected token ID is outside int32 range.")
    if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise ValueError("Selected scores contain NaN or positive infinity.")
    with np.errstate(over="ignore", under="ignore"):
        narrow = scores.astype("<f4")
    # Include signed zero in the exactness check. Negative infinity is valid.
    exact = np.array_equal(narrow.astype("<f8").view("<u8"), scores.view("<u8"))
    encoded = narrow if exact else scores
    return {
        "version": 1,
        "value_dtype": encoded.dtype.str,
        "offsets_b64": _encode(offsets),
        "values_b64": _encode(encoded),
        "ids_b64": _encode(ids.astype("<i4")),
    }


def _decode(encoded: str, dtype: str) -> np.ndarray:
    if not isinstance(encoded, str):
        raise ValueError("Packed selected-score buffers must be base64 strings.")
    try:
        return np.frombuffer(pybase64.b64decode(encoded, validate=True), dtype=dtype)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid packed selected-score buffer.") from exc


def decode_selected_logprobs(payload: dict) -> SelectedLogprobBuffer:
    """Validate once at the HTTP boundary, retaining read-only flat buffers."""
    fields = {"version", "value_dtype", "offsets_b64", "values_b64", "ids_b64"}
    if (not isinstance(payload, dict) or set(payload) != fields
            or type(payload["version"]) is not int or payload["version"] != 1):
        raise ValueError("Unknown packed selected-score schema/version.")
    if payload["value_dtype"] not in ("<f4", "<f8"):
        raise ValueError("Unsupported selected-score value dtype.")
    offsets = _decode(payload["offsets_b64"], "<i8")
    values = _decode(payload["values_b64"], payload["value_dtype"])
    ids = _decode(payload["ids_b64"], "<i4")
    if (not len(offsets) or offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1])
            or offsets[-1] != len(values) or len(values) != len(ids)):
        raise ValueError("Packed selected-score offsets/buffer lengths disagree.")
    if np.any(ids < 0) or np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("Packed selected scores contain invalid values or IDs.")
    return SelectedLogprobBuffer(offsets, values, ids)


def unpack_selected_logprobs(payload: dict, response_length: int) -> list[SelectedLogprobRow]:
    """Compatibility adapter for callers that still hold a wire payload."""
    return decode_selected_logprobs(payload).response_rows(response_length)
