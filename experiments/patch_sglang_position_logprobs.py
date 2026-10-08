"""Install the scoring-only per-target-position selected-ID API on Miles SGLang.

Usage: python patch_sglang_position_logprobs.py /path/to/sglang/python/sglang
Without --write emits an apply_patch document. --write is for remote deployment.
Refuses unexpected source layouts; computes all edits before writing anything.
"""
import argparse
import difflib
from pathlib import Path


def _position_argument_site(text):
    """Recognize the two inspected logprob-processor call layouts, fail closed."""
    matches = []
    for width in (20, 24):
        indent = ' ' * width
        anchor = (f'{indent}token_ids_logprobs_idx,\n'
                  f'{indent}split_len_token_ids,\n'
                  f'{indent}log_normalizer=chunk_log_normalizer,\n')
        if text.count(anchor) == 1:
            addition = (
                f'{indent}position_ids=(logits_metadata.token_ids_logprobs_positions[chunk_slice]\n'
                f'{indent}              if logits_metadata.token_ids_logprobs_positions is not None else None),\n'
            )
            matches.append((anchor, anchor + addition))
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one recognized selected-ID logprob call site')
    return matches[0]


def patch_sources(root):
    changed = {}

    def edit(rel, old, new, count=1):
        path = root / rel
        before, text = changed.get(path, (path.read_text(), path.read_text()))
        if text.count(old) != count:
            raise RuntimeError(f"{rel}: expected {count} matches, found {text.count(old)}: {old[:90]!r}")
        changed[path] = before, text.replace(old, new)

    io = 'srt/managers/io_struct.py'
    edit(io, '    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None\n',
         '    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None\n'
         '    # Selected IDs indexed by target input-token position, not predictor row.\n'
         '    # Single: [position][id]; batch: [request][position][id]. Scoring only.\n'
         '    token_ids_logprob_positions: Optional[List] = None\n')
    edit(io, '    def _normalize_return_hidden_states(self, num):',
         '''        positions = self.token_ids_logprob_positions
        if positions is None:
            self.token_ids_logprob_positions = [None] * num
        elif self.parallel_sample_num != 1 or len(positions) != num:
            raise ValueError("Batched token_ids_logprob_positions must have one entry per request")

    def _normalize_return_hidden_states(self, num):''')
    edit(io, '            token_ids_logprob=self.token_ids_logprob[i],\n',
         '            token_ids_logprob=self.token_ids_logprob[i],\n'
         '            token_ids_logprob_positions=self.token_ids_logprob_positions[i],\n')
    edit(io, '    token_ids_logprob: Optional[List[int]]\n    # Whether to stream output',
         '    token_ids_logprob: Optional[List[int]]\n'
         '    token_ids_logprob_positions: Optional[List[List[int]]] = None\n'
         '    # Whether to stream output')

    tm = 'srt/managers/tokenizer_manager.py'
    edit(tm, '            self._validate_token_ids_logprob(obj)\n',
         '            self._validate_token_ids_logprob(obj)\n'
         '            self._validate_position_logprobs(obj, input_ids)\n')
    edit(tm, '    def _validate_token_ids_logprob(self, obj: GenerateReqInput) -> None:',
         '''    def _validate_position_logprobs(self, obj, input_ids):
        positions = obj.token_ids_logprob_positions
        if positions is None:
            return
        if (not obj.return_logprob or obj.token_ids_logprob is not None
                or obj.sampling_params.get("max_new_tokens") != 0
                or obj.logprob_start_len != 0 or obj.session_params is not None
                or getattr(obj, "multi_item_scoring_delimiter", None) is not None):
            raise ValueError("Per-position IDs require scoring only, logprob_start_len=0, "
                             "and no flat selected IDs, sessions or multi-item scoring")
        if not isinstance(positions, list) or len(positions) != len(input_ids):
            raise ValueError("Per-position IDs must have exactly one row per input token")
        for row in positions:
            if not isinstance(row, list) or any(
                type(token) is not int or not 0 <= token < self.model_config.vocab_size
                for token in row
            ):
                raise ValueError("Per-position IDs contain an invalid row or token ID")

    def _validate_token_ids_logprob(self, obj: GenerateReqInput) -> None:''')
    edit(tm, '                token_ids_logprob=obj.token_ids_logprob,\n',
         '                token_ids_logprob=obj.token_ids_logprob,\n'
         '                token_ids_logprob_positions=obj.token_ids_logprob_positions,\n')
    # The conversion helpers only use this argument as a presence flag. An empty
    # flat list enables selected-logprob output without overloading it with 2D IDs.
    path = root / tm
    count = changed[path][1].count('state.obj.token_ids_logprob,')
    edit(tm, 'state.obj.token_ids_logprob,',
         '([] if state.obj.token_ids_logprob_positions is not None else state.obj.token_ids_logprob),', count)

    sb = 'srt/managers/schedule_batch.py'
    edit(sb, '    token_ids_logprob: Optional[List[int]]\n    input_token_logprobs_val',
         '    token_ids_logprob: Optional[List[int]]\n'
         '    token_ids_logprob_positions: Optional[List[List[int]]] = None\n    input_token_logprobs_val')
    edit(sb, '        token_ids_logprob: List[int] = None,\n',
         '        token_ids_logprob: List[int] = None,\n'
         '        token_ids_logprob_positions: Optional[List[List[int]]] = None,\n')
    edit(sb, '            token_ids_logprob=token_ids_logprob,\n',
         '            token_ids_logprob=token_ids_logprob,\n'
         '            token_ids_logprob_positions=token_ids_logprob_positions,\n')
    for rel in ['srt/managers/scheduler.py', 'srt/session/session_controller.py',
                'srt/disaggregation/encoder/receiver.py']:
        var = 'req' if 'session_controller' in rel else 'recv_req'
        edit(rel, f'            token_ids_logprob={var}.token_ids_logprob,\n',
             f'            token_ids_logprob={var}.token_ids_logprob,\n'
             f'            token_ids_logprob_positions={var}.token_ids_logprob_positions,\n')

    fb = 'srt/model_executor/forward_batch_info.py'
    edit(fb, '    token_ids_logprobs: Optional[List[List[int]]] = None\n',
         '    token_ids_logprobs: Optional[List[List[int]]] = None\n'
         '    # Already aligned to pruned predictor rows for this prefill chunk.\n'
         '    token_ids_logprobs_positions: Optional[List] = None\n')
    edit(fb, '            token_ids_logprobs=batch.token_ids_logprobs,\n',
         '''            token_ids_logprobs=batch.token_ids_logprobs,
            token_ids_logprobs_positions=(
                [
                    None if req.logprob.token_ids_logprob_positions is None else
                    [req.logprob.token_ids_logprob_positions[pos]
                     if pos < len(req.origin_input_ids) else []
                     for pos in range(prefix + start + 1, prefix + length + 1)]
                    for req, prefix, start, length in zip(
                        batch.reqs, batch.prefix_lens,
                        batch.extend_logprob_start_lens, batch.extend_lens, strict=True)
                ] if batch.return_logprob and batch.forward_mode.is_extend() else None
            ),
''')
    edit('srt/model_executor/runner/prefill_cuda_graph_runner.py',
         '            token_ids_logprobs=forward_batch.token_ids_logprobs,\n',
         '            token_ids_logprobs=forward_batch.token_ids_logprobs,\n'
         '            token_ids_logprobs_positions=forward_batch.token_ids_logprobs_positions,\n')

    lm = 'srt/layers/logits_processor.py'
    edit(lm, '    token_ids_logprobs: Optional[List[List[int]]] = None\n',
         '    token_ids_logprobs: Optional[List[List[int]]] = None\n'
         '    token_ids_logprobs_positions: Optional[List] = None\n')
    edit(lm, '                x is not None for x in forward_batch.token_ids_logprobs\n',
         '                x is not None for x in forward_batch.token_ids_logprobs\n'
         '            ) or any(\n'
         '                x is not None for x in (forward_batch.token_ids_logprobs_positions or [])\n')
    edit(lm, '            token_ids_logprobs=forward_batch.token_ids_logprobs,\n',
         '            token_ids_logprobs=forward_batch.token_ids_logprobs,\n'
         '            token_ids_logprobs_positions=forward_batch.token_ids_logprobs_positions,\n')

    lp = 'srt/layers/logprob_processor.py'
    text = (root / lp).read_text()
    begin, end = text.index('def get_token_ids_logprobs_chunk('), text.index('\ndef compute_spec_logprobs(')
    old = text[begin:end]
    new = old.replace('    log_normalizer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,\n',
                      '    log_normalizer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,\n'
                      '    position_ids: Optional[List] = None,\n')
    new = new.replace('    # Empty chunks still walk', '''    if position_ids is not None and any(x is not None for x in position_ids):
        from sglang.srt.layers.position_logprobs import gather_position_chunk
        return gather_position_chunk(
            logprobs, token_ids_logprobs, position_ids, pruned_lens,
            token_ids_logprobs_val, token_ids_logprobs_idx, split_pruned_len,
            log_normalizer)
    # Empty chunks still walk''')
    edit(lp, old, new)
    anchor, replacement = _position_argument_site(text)
    edit(lp, anchor, replacement)
    result = 'srt/managers/scheduler_components/logprob_result_processor.py'
    edit(result, '        if req.logprob.token_ids_logprob is None:\n',
         '        if (req.logprob.token_ids_logprob is None\n'
         '                and req.logprob.token_ids_logprob_positions is None):\n')
    # Input handling only: do not request decode logprobs for position-only scoring.
    text = (root / result).read_text()
    count = text.count('if req.logprob.token_ids_logprob is not None:')
    edit(result, 'if req.logprob.token_ids_logprob is not None:',
         'if (req.logprob.token_ids_logprob is not None\n'
         '                or req.logprob.token_ids_logprob_positions is not None):', count)
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    changed = patch_sources(args.root)
    helper = args.root / 'srt/layers/position_logprobs.py'
    if helper.exists():
        raise RuntimeError(f'Refusing to overwrite existing helper: {helper}')
    changed[helper] = None, Path(__file__).with_name('sglang_position_logprobs.py').read_text()
    if not args.write:
        print('*** Begin Patch')
    for path, (before, after) in changed.items():
        if args.write:
            path.write_text(after)
            print(path)
        else:
            if before is None:
                print(f'*** Add File: {path}')
                for line in after.splitlines():
                    print('+' + line)
                continue
            print(f'*** Update File: {path}')
            diff = list(difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm=''))
            for line in diff[2:]:
                print('@@' if line.startswith('@@') else line)
    if not args.write:
        print('*** End Patch')


if __name__ == '__main__':
    main()
