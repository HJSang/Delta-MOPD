"""Ragged per-position selected-ID gather for scoring-only input logprobs."""
import torch


def gather_position_chunk(logprobs, flat_ids, positions, pruned_lens,
                          output_vals, output_ids, split=0, normalizer=None):
    """Gather once per logit chunk; no per-token GPU synchronization.

    `positions` is already aligned to the pruned predictor rows by ForwardBatch.
    `split` is the number of rows from the first sequence in earlier logit chunks.
    Both chunked prefill and lm-head/logit chunking are therefore supported.
    """
    rows, groups, offset, next_split = [], [], 0, 0
    for n, (ids, selected, length) in enumerate(zip(flat_ids, positions, pruned_lens, strict=True)):
        consumed = split if n == 0 else 0
        remaining = length - consumed
        count = max(0, min(remaining, logprobs.shape[0] - offset))
        if selected is not None:
            if len(selected) != length:
                raise ValueError("Per-position logprob rows do not match pruned length")
            chosen = selected[consumed:consumed + count]
        else:
            chosen = [ids or []] * count
        rows.extend(chosen)
        groups.append((consumed > 0 and remaining > 0, count, ids is not None or selected is not None))
        offset += count
        if count < remaining:
            next_split = consumed + count
            break
    if offset != logprobs.shape[0]:
        raise ValueError("Selected-ID chunk does not cover its logprob rows")

    lengths = [len(row) for row in rows]
    row_indices = [i for i, row in enumerate(rows) for _ in row]
    token_ids = [token for row in rows for token in row]
    if token_ids:
        row_t = torch.tensor(row_indices, device=logprobs.device, dtype=torch.long)
        ids_t = torch.tensor(token_ids, device=logprobs.device, dtype=torch.long)
        values = logprobs[row_t, ids_t]
        if normalizer is not None:
            row_max, row_log_sum = normalizer
            values = (values.float() - row_max[row_t]) - row_log_sum[row_t]
        values = values.tolist()
    else:
        values = []
    ragged, cursor = [], 0
    for length in lengths:
        ragged.append(values[cursor:cursor + length])
        cursor += length
    cursor = 0
    for continuation, count, requested in groups:
        vals = ragged[cursor:cursor + count] if requested else []
        ids = rows[cursor:cursor + count] if requested else []
        if continuation:
            output_vals[-1].extend(vals)
            output_ids[-1].extend(ids)
        else:
            output_vals.append(vals)
            output_ids.append(ids)
        cursor += count
    return next_split
