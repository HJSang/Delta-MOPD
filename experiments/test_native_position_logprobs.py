"""Unit tests for the SGLang patch; run inside the patched Miles image."""
import pytest
import torch

from sglang.srt.layers.position_logprobs import gather_position_chunk
from sglang.srt.managers.io_struct import GenerateReqInput


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('normalized', [False, True])
def test_ragged_gather_across_chunks(device, normalized):
    torch.manual_seed(13)
    raw = torch.randn(9, 29, device=device)
    logits = raw if normalized else raw.log_softmax(-1)
    positions = [[[], [2, 4, 2], [5], [], [7, 8]], None, [[1], [], [3, 4]]]
    flat = [None, [6, 9], None]
    lengths = [5, 1, 3]
    vals, ids = [], []

    def call(start, end, seq_start, seq_end, split):
        norm = (raw[start:end].amax(-1),
                (raw[start:end] - raw[start:end].amax(-1, keepdim=True)).logsumexp(-1)) if normalized else None
        return gather_position_chunk(logits[start:end], flat[seq_start:seq_end],
                                     positions[seq_start:seq_end], lengths[seq_start:seq_end],
                                     vals, ids, split, norm)

    split = call(0, 2, 0, 1, 0)
    assert split == 2
    split = call(2, 7, 0, 3, split)
    assert split == 1
    assert call(7, 9, 2, 3, split) == 0
    expected_ids = [positions[0], [[6, 9]], positions[2]]
    assert ids == expected_ids
    reference = raw.log_softmax(-1)
    row = 0
    for sequence_vals, sequence_ids in zip(vals, ids, strict=True):
        for v, tokens in zip(sequence_vals, sequence_ids, strict=True):
            torch.testing.assert_close(torch.tensor(v), reference[row, tokens].cpu(), atol=1e-6, rtol=1e-6)
            row += 1


def test_empty_rows_and_zero_length_request():
    vals, ids = [], []
    split = gather_position_chunk(torch.zeros(2, 8), [None, None], [[], [[], []]], [0, 2], vals, ids)
    assert split == 0 and ids == [[], [[], []]] and vals == ids


def test_invalid_alignment_rejected():
    with pytest.raises(ValueError, match='pruned length'):
        gather_position_chunk(torch.zeros(2, 8), [None], [[[1]]], [2], [], [])


def test_batched_api_preserves_nested_ids():
    positions = [[[], [1]], [[], [3], [4, 5]]]
    req = GenerateReqInput(input_ids=[[10, 11], [12, 13, 14]],
                           sampling_params={'max_new_tokens': 0}, return_logprob=True,
                           logprob_start_len=0, token_ids_logprob_positions=positions)
    req.normalize_batch_and_arguments()
    assert req[0].token_ids_logprob_positions == positions[0]
    assert req[1].token_ids_logprob_positions == positions[1]


def test_plain_request_unchanged():
    req = GenerateReqInput(input_ids=[[10, 11], [12, 13]])
    req.normalize_batch_and_arguments()
    assert req[0].token_ids_logprob_positions is None
