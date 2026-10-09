"""CPU-only deployment contracts; no engine is launched by these tests."""

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from miles.rollout.paper_opd import _selected_routes
from miles.utils.types import Sample
from recipes.serve import Engine, load_recipe, main, routing_args, server_command, validate_files

ROOT = Path(__file__).resolve().parents[1]


def _config(topology):
    return ROOT / "recipes" / topology / "config.json"


def _edited(tmp_path, topology, edit):
    data = json.loads(_config(topology).read_text())
    edit(data)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("topology,count", [("standalone", 4), ("multi_lora", 2)])
def test_examples_are_valid_and_dry_run_without_sglang(topology, count):
    recipe = load_recipe(_config(topology))
    assert len(recipe.engines) == count
    result = subprocess.run(
        [sys.executable, "-m", "recipes.serve", "--config", str(_config(topology)), "--engine", "anchor"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "CUDA_VISIBLE_DEVICES=0" in result.stdout
    assert "sglang.launch_server" in result.stdout
    assert "--host 127.0.0.1" in result.stdout


def test_standalone_command_snapshot():
    engine = load_recipe(_config("standalone")).engine("math")
    assert server_command(engine) == (
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        "/models/Polaris-7B-Preview",
        "--host",
        "127.0.0.1",
        "--port",
        "31001",
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--context-length",
        "32768",
        "--mem-fraction-static",
        "0.7",
        "--max-running-requests",
        "16",
    )
    multi_gpu = server_command(replace(engine, devices=(1, 2)))
    assert multi_gpu[multi_gpu.index("--tp-size") + 1] == "2"


def test_multilora_command_snapshot_and_base_slot():
    engine = load_recipe(_config("multi_lora")).engine("experts")
    command = server_command(engine)
    assert command[command.index("--enable-lora") :] == (
        "--enable-lora",
        "--lora-backend",
        "triton",
        "--max-loras-per-batch",
        "4",
        "--lora-paths",
        "math=/adapters/qwen3-4b-math",
        "code=/adapters/qwen3-4b-code",
        "if=/adapters/qwen3-4b-if",
    )


@pytest.mark.parametrize("topology", ["standalone", "multi_lora"])
@pytest.mark.parametrize("mode", ["endpoint", "shiftmopd"])
@pytest.mark.parametrize("selection", ["all", "routed"])
def test_routes_preserve_independent_selection_and_target(topology, mode, selection):
    recipe = load_recipe(_config(topology))
    args = routing_args(recipe, target_mode=mode, selection=selection)
    assert args[args.index("--opd-objective") + 1] == "paper"
    assert args[args.index("--opd-target-mode") + 1] == mode
    assert args[args.index("--opd-teacher-selection") + 1] == selection
    assert ("--opd-base-urls" in args) == (mode == "shiftmopd")
    assert ("--opd-teacher-adapters" in args) == (topology == "multi_lora")
    assert "--opd-base-adapters" not in args
    if topology == "multi_lora":
        assert "math=math" in args and "code=code" in args and "if=if" in args
        assert "math=http://127.0.0.1:32001/generate" in args


@pytest.mark.parametrize("topology", ["standalone", "multi_lora"])
@pytest.mark.parametrize("mode", ["endpoint", "shiftmopd"])
@pytest.mark.parametrize("selection", ["all", "routed"])
def test_generated_routes_reach_production_paper_scorer(topology, mode, selection):
    recipe = load_recipe(_config(topology))
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-opd", action="store_true")
    for flag in ("type", "objective", "target-mode", "teacher-selection", "teacher-key", "anchor-url"):
        parser.add_argument(f"--opd-{flag}")
    for flag in ("teacher-urls", "base-urls", "teacher-adapters"):
        parser.add_argument(f"--opd-{flag}", nargs="+")
    args = parser.parse_args(routing_args(recipe, target_mode=mode, selection=selection))
    names, anchor, teachers, bases = _selected_routes(args, Sample(metadata={"opd_teacher": "math"}))
    selected = recipe.teachers if selection == "all" else recipe.teachers[:1]
    assert names == [route.name for route in selected]
    assert teachers == [(recipe.engine(route.engine).url, route.adapter) for route in selected]
    assert anchor == (recipe.engine(recipe.anchor).url, None)
    expected_bases = [(recipe.engine(dict(recipe.bases)[route.name]).url, None) for route in selected]
    assert bases == (expected_bases if mode == "shiftmopd" else [])
    if topology == "multi_lora" and mode == "shiftmopd":
        assert len(set(bases)) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("port", 31000),
        ("port", True),
        ("devices", [0]),
        ("devices", []),
        ("devices", [True]),
        ("devices", [1, 1]),
        ("mem_fraction_static", 1.0),
        ("mem_fraction_static", float("nan")),
        ("context_length", 0),
        ("max_running_requests", -1),
        ("model_path", "relative/model"),
    ],
)
def test_invalid_resource_config_rejected(tmp_path, field, value):
    path = _edited(tmp_path, "standalone", lambda d: d["engines"]["math"].update({field: value}))
    with pytest.raises(ValueError):
        load_recipe(path)


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d["teachers"]["math"].update(adapter="unknown"),
        lambda d: d["bases"].update(math="anchor"),
        lambda d: d["bases"].pop("math"),
        lambda d: d["teachers"].update(code=copy.deepcopy(d["teachers"]["math"])),
        lambda d: d.update(anchor="missing"),
    ],
)
def test_invalid_routes_rejected(tmp_path, edit):
    with pytest.raises(ValueError):
        load_recipe(_edited(tmp_path, "multi_lora", edit))


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"engines": {}, "engines": {}}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_recipe(path)


def _local_assets(tmp_path, declared_base="example/base"):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").touch()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "base_model_name_or_path": declared_base,
            }
        )
    )
    (adapter / "adapter_model.safetensors").touch()
    return Engine("expert", base, "example/base", (1,), 31001, (("math", adapter),))


def test_matching_declared_adapter_base(tmp_path):
    validate_files(_local_assets(tmp_path))


def test_mismatched_declared_adapter_base(tmp_path):
    with pytest.raises(ValueError, match="different base"):
        validate_files(_local_assets(tmp_path, declared_base="other/base"))


def test_missing_model_or_adapter_files(tmp_path):
    engine = _local_assets(tmp_path)
    (engine.adapters[0][1] / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="missing adapter weights"):
        validate_files(engine)
    (engine.model_path / "config.json").unlink()
    with pytest.raises(ValueError, match="Missing model config"):
        validate_files(engine)


def test_execute_uses_argv_and_explicit_gpu_mask(tmp_path, monkeypatch):
    engine = _local_assets(tmp_path)
    config = _edited(tmp_path, "standalone", lambda d: d["engines"]["math"].update(model_path=str(engine.model_path)))
    monkeypatch.setattr(sys, "argv", ["serve", "--config", str(config), "--engine", "math", "--execute"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    called = []
    monkeypatch.setattr("recipes.serve.os.execvpe", lambda exe, argv, env: called.append((exe, argv, env)))
    main()
    assert len(called) == 1
    assert called[0][0] == sys.executable
    assert called[0][2]["CUDA_VISIBLE_DEVICES"] == "1"
    assert called[0][1][1:3] == ("-m", "sglang.launch_server")


def test_execute_cannot_be_combined_with_routes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["serve", "--config", str(_config("standalone")), "--routes", "--execute"])
    with pytest.raises(SystemExit, match="2"):
        main()
