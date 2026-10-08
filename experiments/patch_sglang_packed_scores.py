"""Install lossless selected-score transport after the per-position patch.

No model/scheduler changes. The new request flag affects HTTP output only.
Run without --write for an apply_patch document; stage on a copy before deployment.
Both legacy and raw-array patched tokenizer managers are supported.
"""

import argparse
import difflib
from pathlib import Path


def patch_sources(root: Path) -> dict[Path, tuple[str | None, str]]:
    changed = {}

    def edit(rel, old, new):
        path = root / rel
        before, text = changed.get(path, (path.read_text(), path.read_text()))
        if text.count(old) != 1:
            raise RuntimeError(f"{rel}: expected one matching patch site, got {text.count(old)}")
        changed[path] = before, text.replace(old, new)

    io = "srt/managers/io_struct.py"
    edit(io, "    return_flat_raw_top_logprobs_b64: bool = False\n",
         "    return_flat_raw_top_logprobs_b64: bool = False\n"
         "    # Scoring-only HTTP transport; does not change model computation.\n"
         "    return_packed_token_ids_logprobs: bool = False\n")
    edit(io, "            return_flat_raw_top_logprobs_b64=self.return_flat_raw_top_logprobs_b64,\n",
         "            return_flat_raw_top_logprobs_b64=self.return_flat_raw_top_logprobs_b64,\n"
         "            return_packed_token_ids_logprobs=self.return_packed_token_ids_logprobs,\n")
    tm = "srt/managers/tokenizer_manager.py"
    edit(tm, "import asyncio\n",
         "import asyncio\n"
         "from sglang.srt.utils.selected_logprob_transport import pack_selected_logprobs\n")
    # The original GenerateReqInput remains in ReqState; no flag needs to pass
    # through the scheduler or model forwards. Validate before scheduling.
    edit(tm, "    def _validate_position_logprobs(self, obj, input_ids):\n",
         "    def _validate_position_logprobs(self, obj, input_ids):\n"
         "        if obj.return_packed_token_ids_logprobs and (\n"
         "            obj.stream or obj.sampling_params.get('max_new_tokens') != 0\n"
         "            or not obj.return_logprob or obj.token_ids_logprob_positions is None\n"
         "        ):\n"
         "            raise ValueError('Packed selected scores require non-streaming per-position scoring')\n")
    edit(tm, "        # 3. Handle token_ids_logprob\n",
         "        # 3. Handle token_ids_logprob\n"
         "        if state.obj.return_packed_token_ids_logprobs and token_ids_logprob is not None:\n"
         "            meta_info['input_token_ids_logprobs_packed'] = pack_selected_logprobs(\n"
         "                state.input_token_ids_logprobs_val, state.input_token_ids_logprobs_idx\n"
         "            )\n"
         "            return\n")
    helper = root / "srt/utils/selected_logprob_transport.py"
    if helper.exists():
        raise RuntimeError(f"Refusing to overwrite existing transport helper: {helper}")
    source = Path(__file__).resolve().parents[1] / "miles/utils/selected_logprob_transport.py"
    changed[helper] = None, source.read_text()
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    changed = patch_sources(args.root)  # Validate every site before any write.
    if not args.write:
        print("*** Begin Patch")
    for path, (before, after) in changed.items():
        if args.write:
            path.write_text(after)
            print(path)
        elif before is None:
            print(f"*** Add File: {path}")
            for line in after.splitlines():
                print("+" + line)
        else:
            print(f"*** Update File: {path}")
            for line in list(difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm=""))[2:]:
                print("@@" if line.startswith("@@") else line)
    if not args.write:
        print("*** End Patch")


if __name__ == "__main__":
    main()
