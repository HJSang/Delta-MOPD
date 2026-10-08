"""Correctness-only GRPO rewards using the existing Open-MOPD verifiers.

CPU subprocesses have bounded concurrency, clean environments and a watchdog.
This is process isolation, not an OS security sandbox. Dataset test timeouts
are candidate failures; an outer evaluator crash/timeout is a training error.
"""

import asyncio
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile

_LIMITERS = {}


def correctness(result: dict) -> dict:
    accuracy = float(result["acc"])
    if not math.isfinite(accuracy) or accuracy not in (0.0, 1.0):
        raise ValueError(f"Expected binary verifier correctness, got {accuracy}")
    return {"score": accuracy, "acc": accuracy}


async def _score_one(args, sample):
    domain = sample.metadata["domain"]
    if domain not in ("math", "code"):
        raise ValueError(f"Unexpected GRPO domain: {domain}")
    payload = {"domain": domain, "data_source": sample.metadata["data_source"],
               "response": sample.response, "label": sample.label}
    limiter = _LIMITERS.setdefault(asyncio.get_running_loop(), asyncio.Semaphore(16))
    async with limiter:
        env = {key: os.environ[key] for key in ("PATH", "LANG", "LD_LIBRARY_PATH", "OPEN_MOPD_REWARD_SCORE_ROOT") if key in os.environ}
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="grpo-verifier-") as directory:
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(Path(__file__).resolve()), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=env, cwd=directory, start_new_session=True)
            task = asyncio.create_task(process.communicate(json.dumps(payload).encode()))
            try:
                stdout, stderr = await asyncio.wait_for(asyncio.shield(task), timeout=180)
            except (TimeoutError, asyncio.CancelledError):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await task
                raise
            if process.returncode:
                raise RuntimeError(f"GRPO verifier failed: {stderr.decode(errors='replace')[-3000:]}")
            return correctness(json.loads(stdout.decode().splitlines()[-1]))


async def reward_func(args, samples, **kwargs):
    del kwargs
    # Miles calls global custom rewards with a list, eval overrides with a sample.
    if isinstance(samples, list):
        return await asyncio.gather(*(_score_one(args, sample) for sample in samples))
    return await _score_one(args, samples)


if __name__ == "__main__":
    # Direct child execution avoids importing Miles/torch into CPU verifiers.
    from open_mopd import compute_open_mopd_score

    payload = json.load(sys.stdin)
    result = compute_open_mopd_score(
        data_source=payload["data_source"], solution_str=payload["response"],
        ground_truth=payload["label"], scorer=payload["domain"],
        strict_box_verify=True, max_tests=15, timeout=6, testcase_max_workers=1)
    print(json.dumps(correctness(result)))
