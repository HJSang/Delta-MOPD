"""Bounded, vectorized ShiftMOPD composition with a legacy-map adapter."""
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from miles.utils.opd_centering import center_sampled_rewards
from miles.utils.selected_logprobs import selected_logprob_rows
from miles.utils.selected_logprob_transport import SelectedLogprobRow


@dataclass(frozen=True)
class SelectedLogprobRows:
    """View of scorer rows; do not allocate a Python dict for every token."""

    rows: list

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _aligned_tile(rows, supports, start, width):
    # Zero padding is ignored until the final support mask, avoiding inf-inf.
    result = np.zeros((len(supports), width), dtype=np.float64)
    for offset, ids in enumerate(supports):
        row = rows[start + offset]
        if isinstance(row, SelectedLogprobRow) and np.array_equal(row.token_ids, ids):
            result[offset, :len(ids)] = row.values
            continue
        if not isinstance(row, Mapping):
            entries = row or []
            row_ids = [int(entry[1]) for entry in entries]
            if row_ids == ids:
                result[offset, :len(ids)] = [entry[0] for entry in entries]
                continue
            row = {int(entry[1]): float(entry[0]) for entry in entries}
        try:
            result[offset, :len(ids)] = [row[token_id] for token_id in ids]
        except KeyError as exc:
            raise ValueError(f"ShiftMOPD scorer response is missing token {exc.args[0]} at position {start + offset}.") from exc
    return result


def compose_reverse_kl(anchor, teachers, bases, student, supports, *, loss_mask=None,
                       center=True, tile_size=2048, timings=None):
    """Preserve Python FP64 accumulation order, FP32 partition, and response mean.

    Distinct scorer objects are converted once per tile (shared bases reused).
    Tiles bound additional numeric memory; they do not split model forwards or
    change the scored prefix. A single mean is subtracted after all tiles.
    """
    start_time = perf_counter()
    if not teachers:
        raise ValueError("ShiftMOPD requires at least one teacher.")
    if len(teachers) != len(bases):
        raise ValueError("ShiftMOPD teacher/base count mismatch.")
    n = len(supports)
    if student.ndim != 1 or student.numel() != n or any(len(x) != n for x in [anchor, *teachers, *bases]):
        raise ValueError("ShiftMOPD scorer lengths do not match the response length.")
    if tile_size <= 0:
        raise ValueError("ShiftMOPD tile size must be positive.")
    if any(not ids or len(set(ids)) != len(ids) for ids in supports):
        raise ValueError("ShiftMOPD support is empty or contains duplicate IDs.")
    raw = torch.empty(n, dtype=torch.float32, device=student.device)
    align_seconds = math_seconds = 0.0
    for start in range(0, n, tile_size):
        chosen = supports[start:start + tile_size]
        lengths = np.asarray([len(ids) for ids in chosen])
        width = int(lengths.max())
        before = perf_counter()
        arrays = {id(x): _aligned_tile(x, chosen, start, width) for x in
                  {id(x): x for x in [anchor, *teachers, *bases]}.values()}
        align_seconds += perf_counter() - before
        before = perf_counter()
        scores = arrays[id(anchor)].copy()
        for teacher, base in zip(teachers, bases, strict=True):
            # Do not replace with sum(teachers)-3*base: preserve operation order.
            scores += arrays[id(teacher)] - arrays[id(base)]
        scores[np.arange(width)[None, :] >= lengths[:, None]] = -np.inf
        values = torch.from_numpy(scores.astype(np.float32))
        sampled = values[torch.arange(len(chosen)), torch.from_numpy(lengths - 1)]
        target = (sampled - torch.logsumexp(values, dim=-1)).to(student.device)
        raw[start:start + len(chosen)] = student[start:start + len(chosen)].float() - target
        math_seconds += perf_counter() - before
    if not torch.isfinite(raw).all():
        raise ValueError("ShiftMOPD produced a non-finite sampled reverse-KL.")
    centered, _ = center_sampled_rewards(raw, loss_mask, center=center)
    if timings is not None:
        timings.update(alignment_seconds=align_seconds, composition_seconds=math_seconds,
                       postprocess_seconds=perf_counter()-start_time,
                       distinct_scorers=len({id(x) for x in [anchor, *teachers, *bases]}),
                       support_entries=sum(map(len, supports)))
    return centered, raw


def compose_scorer_responses(reward, student, *, loss_mask=None, center=True,
                             tile_size=2048, reference=None):
    """Parse each shared HTTP response once, retaining JSON-row views when fast."""
    start = perf_counter()
    n = student.numel()
    names = list(reward["teachers"])
    responses = [reward["anchor"], *(reward["teachers"][k] for k in names),
                 *(reward["bases"][k] for k in names)]
    unique = {id(response): response for response in responses}
    parsed = {}
    timings = {}
    for key, response in unique.items():
        meta_info = response["meta_info"]
        rows = selected_logprob_rows(meta_info, n)
        parsed[key] = ([{int(entry[1]): float(entry[0]) for entry in row or []} for row in rows]
                       if reference else SelectedLogprobRows(rows))
        for metric, value in response.get("_opd_transport", {}).items():
            timings[metric] = timings.get(metric, 0.) + value
    timings["response_prepare_seconds"] = perf_counter()-start
    anchor = parsed[id(reward["anchor"])]
    teachers = [parsed[id(reward["teachers"][name])] for name in names]
    bases = [parsed[id(reward["bases"][name])] for name in names]
    kwargs = dict(loss_mask=loss_mask, center=center)
    if reference is None:
        kwargs.update(tile_size=tile_size, timings=timings)
    centered, raw = (reference or compose_reverse_kl)(anchor, teachers, bases, student,
                                                     reward["support_ids"], **kwargs)
    timings["response_total_seconds"] = perf_counter()-start
    return centered, raw, timings
