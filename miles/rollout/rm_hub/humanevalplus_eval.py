"""Official EvalPlus base+plus accuracy, outside the rollout event loop.

Each answer is checked in a separate CPU subprocess with EvalPlus's execution
guard/time limits. This is process isolation, not an OS security sandbox.
No training reward or loss path imports this evaluator.
"""

import asyncio
import faulthandler
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import uuid

_LIMITERS = {}


def score(payload):
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
    from evalplus.evaluate import check_correctness, get_groundtruth
    from evalplus.sanitize import sanitize

    tasks = get_human_eval_plus()
    dataset_hash = get_human_eval_plus_hash()
    if payload["evalplus_hash"] != dataset_hash:
        raise ValueError("EvalPlus dataset hash differs from frozen evaluation data")
    task = tasks[payload["task_id"]]
    expected = get_groundtruth(tasks, dataset_hash, [])
    solution = sanitize(payload["response"], entrypoint=task["entry_point"])
    result = check_correctness("humaneval", 0, task, solution, expected[task["task_id"]], fast_check=True)
    base = result["base"][0] == "pass"
    plus = result["plus"][0] == "pass"
    return {"score": float(base and plus), "acc": float(base and plus), "base_acc": float(base),
            "plus_acc": float(plus), "base_status": result["base"][0], "plus_status": result["plus"][0]}


async def reward_func(args, sample, **kwargs):
    del kwargs
    payload = {"task_id": sample.metadata["task_id"],
               "evalplus_hash": sample.metadata["evalplus_hash"], "response": sample.response}
    limiter = _LIMITERS.setdefault(asyncio.get_running_loop(), asyncio.Semaphore(8))
    async with limiter:
        # Do not forward W&B/API tokens to generated-code execution processes.
        env = {key: os.environ[key] for key in ("PATH", "LANG", "LD_LIBRARY_PATH") if key in os.environ}
        env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        with tempfile.TemporaryDirectory(prefix="miles-evalplus-") as directory:
            process = await asyncio.create_subprocess_exec(sys.executable, str(Path(__file__).resolve()),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=env, cwd=directory, start_new_session=True)
            communication = asyncio.create_task(process.communicate(json.dumps(payload).encode()))
            try:
                # Official base and plus execution can each consume ~62 seconds.
                # Keep independent headroom for imports and long-answer extraction;
                # this watchdog does not change EvalPlus's execution time limits.
                stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout=300)
            except (TimeoutError, asyncio.CancelledError) as exc:
                # Kill descendants as well, even if the direct child has exited.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = await communication
                if isinstance(exc, TimeoutError):
                    failure = {**payload, "error": "evaluator_timeout", "stderr": stderr.decode(errors="replace")}
                    if getattr(args, "save", None):
                        folder = Path(args.save).parent / "eval_failures"
                        folder.mkdir(parents=True, exist_ok=True)
                        (folder / f"timeout_{uuid.uuid4().hex}.json").write_text(json.dumps(failure))
                    # Infrastructure failures are not fabricated accuracy=0 scores.
                    raise RuntimeError(f"EvalPlus evaluator timeout: {payload['task_id']}; "
                                       f"{failure['stderr'][-4000:]}") from exc
                raise
            if process.returncode:
                raise RuntimeError(f"EvalPlus evaluator failed: {stderr.decode()[-2000:]}")
            return json.loads(stdout.decode().splitlines()[-1])


if __name__ == "__main__":
    import evalplus.sanitize
    from evalplus_extract import code_extract

    # Scoped to this isolated evaluator process. Replace only the exhaustive
    # search with its parity-tested equivalent; retain official sanitize and tests.
    evalplus.sanitize.code_extract = code_extract
    faulthandler.dump_traceback_later(270)
    print(json.dumps(score(json.load(sys.stdin))))
