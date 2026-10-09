"""Launch one frozen SGLang engine, or print Miles routing arguments.

Examples (from the repository root):
    python -m recipes.serve --config recipes/standalone/config.json --engine anchor
    python -m recipes.serve --config recipes/multi_lora/config.json --routes

The default only prints a command. --execute validates local model/adapter
files and replaces this process with SGLang. No downloads, background jobs,
cluster allocation, or process termination are performed by this module.
"""

import argparse
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Engine:
    name: str
    model_path: Path
    model_id: str
    devices: tuple[int, ...]
    port: int
    adapters: tuple[tuple[str, Path], ...] = ()
    context_length: int = 32768
    mem_fraction_static: float = 0.7
    max_running_requests: int = 16

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/generate"


@dataclass(frozen=True)
class Route:
    name: str
    engine: str
    adapter: str | None = None


@dataclass(frozen=True)
class Recipe:
    engines: tuple[Engine, ...]
    anchor: str
    teachers: tuple[Route, ...]
    bases: tuple[tuple[str, str], ...]

    def engine(self, name: str) -> Engine:
        for engine in self.engines:
            if engine.name == name:
                return engine
        raise ValueError(f"Unknown engine: {name}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", value) or value == "none":
        raise ValueError(f"Invalid logical name: {value!r}")
    return value


def load_recipe(path: Path) -> Recipe:
    data = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    if set(data) != {"engines", "anchor", "teachers", "bases"}:
        raise ValueError("Recipe requires exactly engines, anchor, teachers, and bases.")
    engines = []
    for name, raw in data["engines"].items():
        options = dict(raw)
        adapters = tuple((_name(k), Path(v)) for k, v in options.pop("adapters", {}).items())
        engines.append(
            Engine(
                name=_name(name),
                model_path=Path(options.pop("model_path")),
                devices=tuple(options.pop("devices")),
                adapters=adapters,
                **options,
            )
        )
    recipe = Recipe(
        engines=tuple(engines),
        anchor=data["anchor"],
        teachers=tuple(Route(name=_name(k), **v) for k, v in data["teachers"].items()),
        bases=tuple(data["bases"].items()),
    )
    validate_recipe(recipe)
    return recipe


def validate_recipe(recipe: Recipe) -> None:
    ports, devices = set(), set()
    if not recipe.engines or not recipe.teachers:
        raise ValueError("At least one engine and teacher are required.")
    for engine in recipe.engines:
        if not engine.model_path.is_absolute() or not engine.model_id:
            raise ValueError(f"{engine.name}: use an absolute local model path and source model ID.")
        if type(engine.port) is not int or not 1024 <= engine.port <= 65535 or engine.port in ports:
            raise ValueError("Engine ports must be unique integers in [1024, 65535].")
        ports.add(engine.port)
        if (
            not engine.devices
            or any(type(d) is not int or d < 0 for d in engine.devices)
            or len(set(engine.devices)) != len(engine.devices)
            or devices.intersection(engine.devices)
        ):
            raise ValueError("Engine GPU indices must be nonnegative, unique, and disjoint.")
        devices.update(engine.devices)
        if not 0 < engine.mem_fraction_static < 1:
            raise ValueError("mem_fraction_static must lie strictly between 0 and 1.")
        if (
            type(engine.context_length) is not int
            or engine.context_length <= 0
            or type(engine.max_running_requests) is not int
            or engine.max_running_requests <= 0
        ):
            raise ValueError("Context length and concurrency must be positive integers.")
        if any(not path.is_absolute() for _, path in engine.adapters):
            raise ValueError("Adapter paths must be absolute local directories.")
    recipe.engine(recipe.anchor)
    if {r.name for r in recipe.teachers} != dict(recipe.bases).keys():
        raise ValueError("Teacher and precursor names must match exactly.")
    seen = set()
    for route in recipe.teachers:
        engine = recipe.engine(route.engine)
        recipe.engine(dict(recipe.bases)[route.name])
        if route.adapter is not None and route.adapter not in dict(engine.adapters):
            raise ValueError(f"{route.name}: adapter is not registered on its engine.")
        if route.adapter is not None and dict(recipe.bases)[route.name] != route.engine:
            raise ValueError("A LoRA teacher's precursor must be that engine's unadapted base.")
        identity = (route.engine, route.adapter)
        if identity in seen:
            raise ValueError("Do not count the same teacher route twice; use one logical route per expert.")
        seen.add(identity)


def validate_files(engine: Engine) -> None:
    """Check local assets and declared PEFT lineage, not numerical equivalence."""
    if not (engine.model_path / "config.json").is_file():
        raise ValueError(f"Missing model config: {engine.model_path}")
    if not any(engine.model_path.glob("*.safetensors")) and not any(engine.model_path.glob("pytorch_model*.bin")):
        raise ValueError(f"No HF model weights found: {engine.model_path}")
    for name, path in engine.adapters:
        config = json.loads((path / "adapter_config.json").read_text())
        if config.get("peft_type") != "LORA" or not isinstance(config.get("r"), int) or config["r"] <= 0:
            raise ValueError(f"{name}: expected a positive-rank PEFT LoRA adapter.")
        base = config.get("base_model_name_or_path", "")
        same_path = bool(base) and Path(base).is_absolute() and Path(base).resolve() == engine.model_path.resolve()
        if base != engine.model_id and not same_path:
            raise ValueError(
                f"{name}: adapter declares a different base; do not override its metadata to bypass this."
            )
        if not any((path / filename).is_file() for filename in ("adapter_model.safetensors", "adapter_model.bin")):
            raise ValueError(f"{name}: missing adapter weights.")


def server_command(engine: Engine) -> tuple[str, ...]:
    command = (
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(engine.model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(engine.port),
        "--tp-size",
        str(len(engine.devices)),
        "--dtype",
        "bfloat16",
        "--context-length",
        str(engine.context_length),
        "--mem-fraction-static",
        str(engine.mem_fraction_static),
        "--max-running-requests",
        str(engine.max_running_requests),
    )
    if engine.adapters:
        command += (
            "--enable-lora",
            "--lora-backend",
            "triton",
            "--max-loras-per-batch",
            str(len(engine.adapters) + 1),
            "--lora-paths",
            *(f"{name}={path}" for name, path in engine.adapters),
        )
    return command


def routing_args(recipe: Recipe, *, target_mode: str, selection: str) -> tuple[str, ...]:
    if target_mode not in ("endpoint", "shiftmopd") or selection not in ("all", "routed"):
        raise ValueError("Choose endpoint|shiftmopd and all|routed.")
    args = (
        "--use-opd",
        "--opd-type",
        "sglang",
        "--opd-objective",
        "paper",
        "--opd-target-mode",
        target_mode,
        "--opd-teacher-selection",
        selection,
        "--opd-teacher-key",
        "opd_teacher",
        "--opd-anchor-url",
        recipe.engine(recipe.anchor).url,
        "--opd-teacher-urls",
        *(f"{r.name}={recipe.engine(r.engine).url}" for r in recipe.teachers),
    )
    if any(r.adapter is not None for r in recipe.teachers):
        args += ("--opd-teacher-adapters", *(f"{r.name}={r.adapter or 'none'}" for r in recipe.teachers))
    if target_mode == "shiftmopd":
        args += ("--opd-base-urls", *(f"{name}={recipe.engine(engine).url}" for name, engine in recipe.bases))
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--engine")
    action.add_argument("--routes", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Validate assets and start the selected engine")
    parser.add_argument("--target-mode", choices=("endpoint", "shiftmopd"), default="shiftmopd")
    parser.add_argument("--selection", choices=("all", "routed"), default="all")
    args = parser.parse_args()
    if args.routes and args.execute:
        parser.error("--execute requires --engine, not --routes")
    recipe = load_recipe(args.config)
    if args.routes:
        print(shlex.join(routing_args(recipe, target_mode=args.target_mode, selection=args.selection)))
        return
    engine = recipe.engine(args.engine)
    command = server_command(engine)
    gpu_mask = ",".join(map(str, engine.devices))
    print(shlex.join((f"CUDA_VISIBLE_DEVICES={gpu_mask}", *command)), flush=True)
    if args.execute:
        validate_files(engine)
        os.execvpe(command[0], command, {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_mask})


if __name__ == "__main__":
    main()
