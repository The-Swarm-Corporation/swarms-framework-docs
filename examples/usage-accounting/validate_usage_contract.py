#!/usr/bin/env python3
"""Validate Swarms token-accounting contracts without provider calls.

This script uses only the Python standard library. It reads a Swarms source
checkout, extracts the accounting functions and methods from its AST, and
executes those exact method bodies inside small dependency-free shells.

Usage:
    python examples/usage-accounting/validate_usage_contract.py /path/to/swarms
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional


USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "total_tokens",
)


def parse_source(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def top_level_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return copy.deepcopy(node)
    raise AssertionError(f"missing function: {name}")


def class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class: {name}")


def class_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return copy.deepcopy(node)
    raise AssertionError(f"missing method: {cls.name}.{name}")


def compile_functions(
    functions: list[ast.FunctionDef], namespace: dict
) -> dict:
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "<extracted-functions>", "exec"), namespace)
    return namespace


def compile_methods(
    methods: list[ast.FunctionDef], namespace: dict, name: str
):
    node = ast.ClassDef(
        name=name,
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "<extracted-methods>", "exec"), namespace)
    return namespace[name]


def git_head(source_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def validate_normalization(utils_tree: ast.Module) -> None:
    namespace = {"Optional": Optional}
    compile_functions(
        [
            top_level_function(utils_tree, "_field"),
            top_level_function(utils_tree, "empty_usage"),
            top_level_function(utils_tree, "usage_from_response"),
        ],
        namespace,
    )

    assert tuple(namespace["empty_usage"]().keys()) == USAGE_KEYS

    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=40,
            completion_tokens=18,
            total_tokens=58,
            prompt_tokens_details=SimpleNamespace(cached_tokens=15),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=7),
        )
    )
    usage = namespace["usage_from_response"](response)
    assert usage == {
        "input_tokens": 40,
        "output_tokens": 18,
        "cached_tokens": 15,
        "reasoning_tokens": 7,
        "total_tokens": 58,
    }
    assert usage["cached_tokens"] <= usage["input_tokens"]
    assert usage["reasoning_tokens"] <= usage["output_tokens"]

    response.usage.total_tokens = 0
    fallback = namespace["usage_from_response"](response)
    assert fallback["total_tokens"] == 58


def validate_agent_properties(agent_tree: ast.Module) -> None:
    agent_cls = class_node(agent_tree, "Agent")

    def count_tokens(text: str, model: str | None = None) -> int:
        del model
        return len(text.split())

    ExtractedAgent = compile_methods(
        [
            class_method(agent_cls, "input_tokens"),
            class_method(agent_cls, "usage"),
            class_method(agent_cls, "_add_usage"),
        ],
        {"json": json, "count_tokens": count_tokens},
        "ExtractedAgent",
    )

    class Memory:
        def __init__(self, text: str):
            self.text = text

        def return_history_as_string(self) -> str:
            return self.text

    agent = ExtractedAgent()
    agent.short_memory = Memory("System: terse\nUser: hello")
    agent.tools_list_dictionary = []
    agent.model_name = "gpt-5.4"
    agent._usage = {key: 0 for key in USAGE_KEYS}

    base_estimate = agent.input_tokens
    agent.short_memory.text += "\nUser: " + "context " * 100
    assert agent.input_tokens > base_estimate

    without_tools = agent.input_tokens
    agent.tools_list_dictionary = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {}},
        }
    ]
    assert agent.input_tokens > without_tools

    snapshot = agent.usage
    snapshot["input_tokens"] = 999
    assert agent.usage["input_tokens"] == 0

    agent._add_usage(
        {
            "input_tokens": 12,
            "output_tokens": 3,
            "cached_tokens": 5,
            "reasoning_tokens": 1,
            "total_tokens": 15,
        }
    )
    assert agent.usage["total_tokens"] == 15


def validate_streaming(utils_tree: ast.Module) -> None:
    namespace = {"Optional": Optional}
    compile_functions(
        [
            top_level_function(utils_tree, "_field"),
            top_level_function(utils_tree, "empty_usage"),
            top_level_function(utils_tree, "usage_from_response"),
        ],
        namespace,
    )

    lite_cls = class_node(utils_tree, "LiteLLM")
    ExtractedLiteLLM = compile_methods(
        [
            class_method(lite_cls, "_accumulate_usage"),
            class_method(lite_cls, "_track_streaming_usage"),
        ],
        namespace,
        "ExtractedLiteLLM",
    )

    tracker = ExtractedLiteLLM()
    tracker.usage = namespace["empty_usage"]()
    seen = []
    tracker.usage_hook = seen.append

    content = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))],
        usage=None,
    )
    final = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            prompt_tokens_details=None,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ),
    )

    forwarded = list(tracker._track_streaming_usage(iter([content, final])))
    assert forwarded == [content]
    assert tracker.usage == {
        "input_tokens": 10,
        "output_tokens": 4,
        "cached_tokens": 0,
        "reasoning_tokens": 2,
        "total_tokens": 14,
    }
    assert seen == [tracker.usage]

    build = class_method(lite_cls, "_build_completion_params")
    dump = ast.dump(build, include_attributes=False)
    assert "stream_options" in dump
    assert "include_usage" in dump


def validate_router(router_tree: ast.Module) -> None:
    class FakeAgent:
        def __init__(self, usage: dict):
            self.usage = usage

    def empty_usage() -> dict:
        return {key: 0 for key in USAGE_KEYS}

    router_cls = class_node(router_tree, "SwarmRouter")
    ExtractedRouter = compile_methods(
        [
            class_method(router_cls, "usage"),
            class_method(router_cls, "_usage_agents"),
        ],
        {"Agent": FakeAgent, "List": List, "empty_usage": empty_usage},
        "ExtractedRouter",
    )

    worker = FakeAgent(
        {
            "input_tokens": 100,
            "output_tokens": 10,
            "cached_tokens": 20,
            "reasoning_tokens": 3,
            "total_tokens": 110,
        }
    )
    director = FakeAgent(
        {
            "input_tokens": 30,
            "output_tokens": 5,
            "cached_tokens": 0,
            "reasoning_tokens": 1,
            "total_tokens": 35,
        }
    )

    router = ExtractedRouter()
    router.agents = [worker]
    router._swarm_cache = {
        "built": SimpleNamespace(
            agents=[worker],
            director=director,
            conversation=None,
        )
    }

    assert router.usage == {
        "input_tokens": 130,
        "output_tokens": 15,
        "cached_tokens": 20,
        "reasoning_tokens": 4,
        "total_tokens": 145,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Swarms usage-accounting source contracts offline."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to a kyegomez/swarms source checkout",
    )
    args = parser.parse_args()

    root = args.source.resolve()
    files = {
        "agent": root / "swarms" / "structs" / "agent.py",
        "router": root / "swarms" / "structs" / "swarm_router.py",
        "utils": root / "swarms" / "utils" / "litellm_wrapper.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        parser.error("missing source files: " + ", ".join(missing))

    trees = {name: parse_source(path) for name, path in files.items()}

    checks = [
        (
            "normalized usage fields",
            lambda: validate_normalization(trees["utils"]),
        ),
        (
            "Agent usage and input estimate",
            lambda: validate_agent_properties(trees["agent"]),
        ),
        (
            "stream-final accounting",
            lambda: validate_streaming(trees["utils"]),
        ),
        (
            "SwarmRouter aggregation and dedupe",
            lambda: validate_router(trees["router"]),
        ),
    ]

    print(f"source: {root}")
    print(f"commit: {git_head(root)}")
    for label, check in checks:
        check()
        print(f"PASS: {label}")

    print(f"PASS: {len(checks)} offline contract checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
