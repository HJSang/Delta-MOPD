"""Require a patched scorer and compare legacy/packed HTTP scores exactly.

Use an isolated scoring server with identical execution shapes (no live training
traffic), e.g. with radix cache disabled. This never accepts silent fallback.
"""

import argparse
import json
from urllib.request import Request, urlopen

import numpy as np

from miles.utils.selected_logprob_transport import FIELD, REQUEST_FIELD
from miles.utils.selected_logprobs import selected_logprob_rows


def post(url, payload):
    request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as result:
        body = result.read()
    return json.loads(body), len(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--adapters", nargs="+", default=["none", "math", "code"])
    parser.add_argument("--positions", type=int, default=128)
    args = parser.parse_args()
    # Empty prompt rows, ragged supports, repeats across rows and selected
    # sampled IDs exercise the actual position-indexed GPU gather.
    prompt = [151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]
    tokens = prompt + [40 + i % 100 for i in range(args.positions)]
    ids = [[] for _ in prompt] + [list(range(128 + i % 2)) for i in range(args.positions)]
    payload = dict(input_ids=tokens, token_ids_logprob_positions=ids,
                   sampling_params=dict(temperature=0, max_new_tokens=0),
                   return_logprob=True, logprob_start_len=0)
    for adapter in args.adapters:
        request = {**payload, **({"lora_path": adapter} if adapter != "none" else {})}
        outputs = []
        for packed in (False, True):
            result, size = post(args.url, {**request, REQUEST_FIELD: packed})
            assert (FIELD in result["meta_info"]) == packed, "Scorer did not honor transport request"
            rows = selected_logprob_rows(result["meta_info"], args.positions)
            outputs.append((rows, result, size))
        for left, right in zip(outputs[0][0], outputs[1][0], strict=True):
            left, right = list(left), list(right)
            assert [x[1] for x in left] == [x[1] for x in right]
            a = np.array([x[0] for x in left], dtype="<f8")
            b = np.array([x[0] for x in right], dtype="<f8")
            assert np.array_equal(a.view("<u8"), b.view("<u8")), "Scorer values changed"
        assert outputs[0][1]["meta_info"]["input_token_logprobs"] == outputs[1][1]["meta_info"]["input_token_logprobs"]
        print(json.dumps(dict(adapter=adapter, positions=args.positions, max_abs_error=0,
                              nested_bytes=outputs[0][2], packed_bytes=outputs[1][2],
                              value_dtype=outputs[1][1]["meta_info"][FIELD]["value_dtype"])), flush=True)


if __name__ == "__main__":
    main()
