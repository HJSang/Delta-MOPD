"""Add lossless compact selected-token logprob responses to patched SGLang.

Apply this after ``patch_sglang_position_logprobs.py``.  The default response
contains ``[value, token_id, text]`` triples.  MOPD does not use text, so this
opt-in flag returns parallel ragged value/id arrays instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def patch_sources(root: Path) -> dict[Path, tuple[str | None, str]]:
    changed: dict[Path, tuple[str | None, str]] = {}

    def edit(rel: str, old: str, new: str, count: int = 1) -> None:
        path = root / rel
        before, text = changed.get(path, (path.read_text(), path.read_text()))
        if text.count(old) != count:
            raise RuntimeError(
                f"{rel}: expected {count} matches, found {text.count(old)}: {old[:100]!r}"
            )
        changed[path] = before, text.replace(old, new)

    io = "srt/managers/io_struct.py"
    edit(
        io,
        "    return_flat_raw_top_logprobs_b64: bool = False\n",
        "    return_flat_raw_top_logprobs_b64: bool = False\n"
        "    # Return selected input-token logprobs as parallel raw value/id arrays.\n"
        "    return_raw_token_ids_logprobs: bool = False\n",
    )
    edit(
        io,
        "            return_flat_raw_top_logprobs_b64=self.return_flat_raw_top_logprobs_b64,\n",
        "            return_flat_raw_top_logprobs_b64=self.return_flat_raw_top_logprobs_b64,\n"
        "            return_raw_token_ids_logprobs=self.return_raw_token_ids_logprobs,\n",
    )
    edit(
        io,
        "    return_flat_raw_top_logprobs: bool = False\n\n    # Whether to return hidden states",
        "    return_flat_raw_top_logprobs: bool = False\n"
        "    return_raw_token_ids_logprobs: bool = False\n\n"
        "    # Whether to return hidden states",
    )

    tm = "srt/managers/tokenizer_manager.py"
    edit(
        tm,
        "                return_flat_raw_top_logprobs=obj.return_flat_raw_top_logprobs,\n",
        "                return_flat_raw_top_logprobs=obj.return_flat_raw_top_logprobs,\n"
        "                return_raw_token_ids_logprobs=obj.return_raw_token_ids_logprobs,\n",
    )

    sb = "srt/managers/schedule_batch.py"
    edit(
        sb,
        "    token_ids_logprob_positions: Optional[List[List[int]]] = None\n",
        "    token_ids_logprob_positions: Optional[List[List[int]]] = None\n"
        "    return_raw_token_ids_logprobs: bool = False\n",
    )
    edit(
        sb,
        "        return_flat_raw_top_logprobs: bool = False,\n",
        "        return_flat_raw_top_logprobs: bool = False,\n"
        "        return_raw_token_ids_logprobs: bool = False,\n",
    )
    edit(
        sb,
        "        self.return_flat_raw_top_logprobs = return_flat_raw_top_logprobs\n",
        "        self.return_flat_raw_top_logprobs = return_flat_raw_top_logprobs\n"
        "        self.return_raw_token_ids_logprobs = return_raw_token_ids_logprobs\n",
    )
    for rel, var in [
        ("srt/managers/scheduler.py", "recv_req"),
        ("srt/session/session_controller.py", "req"),
        ("srt/disaggregation/encoder/receiver.py", "recv_req"),
    ]:
        edit(
            rel,
            f"            token_ids_logprob_positions={var}.token_ids_logprob_positions,\n",
            f"            token_ids_logprob_positions={var}.token_ids_logprob_positions,\n"
            f"            return_raw_token_ids_logprobs={var}.return_raw_token_ids_logprobs,\n",
        )

    # Preserve raw arrays and skip the text-producing detokenization branch.
    edit(
        tm,
        "        # 3. Handle token_ids_logprob\n        if token_ids_logprob is not None:\n",
        "        # 3. Handle token_ids_logprob\n"
        "        raw_token_ids = getattr(state.obj, 'return_raw_token_ids_logprobs', False)\n"
        "        if token_ids_logprob is not None:\n",
    )
    edit(
        tm,
        "            if len(state.input_token_ids_logprobs_val) > len(\n                state.input_token_ids_logprobs\n            ):\n",
        "            if (not raw_token_ids and\n"
        "                    len(state.input_token_ids_logprobs_val) > len(\n"
        "                        state.input_token_ids_logprobs\n"
        "                    )):\n",
    )
    edit(
        tm,
        "            meta_info[\"input_token_ids_logprobs\"] = state.input_token_ids_logprobs\n",
        "            if raw_token_ids:\n"
        "                meta_info[\"input_token_ids_logprobs_val\"] = state.input_token_ids_logprobs_val\n"
        "                meta_info[\"input_token_ids_logprobs_idx\"] = state.input_token_ids_logprobs_idx\n"
        "            else:\n"
        "                meta_info[\"input_token_ids_logprobs\"] = state.input_token_ids_logprobs\n",
    )
    edit(
        tm,
        "            meta_info[\"output_token_ids_logprobs\"] = state.output_token_ids_logprobs\n",
        "            if not raw_token_ids:\n"
        "                meta_info[\"output_token_ids_logprobs\"] = state.output_token_ids_logprobs\n",
    )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    changed = patch_sources(args.root)
    for path, (before, after) in changed.items():
        if args.write:
            path.write_text(after)
            print(path)
        else:
            print(f"{path}: {len(before or '')} -> {len(after)} bytes")


if __name__ == "__main__":
    main()
