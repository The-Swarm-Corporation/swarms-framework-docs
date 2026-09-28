"""Scripted, Agent-compatible participants for the offline AuctionSwarm tutorials.

These replace model-backed agents; they do not exercise Agent or a provider SDK.
AuctionSwarm, its concurrent execution helper, and its history formatter are real.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScriptedAgent:
    agent_name: str
    confidence: float
    estimated_cost: float
    fail_on_execute: bool = False
    tools_list_dictionary: list[dict[str, Any]] = field(default_factory=list)
    short_memory: list[str] = field(default_factory=list)
    executed_tasks: list[str] = field(default_factory=list)
    llm: Any = None

    def llm_handling(self) -> None:
        # AuctionSwarm rebuilds the client when it changes the tool list.
        # This stand-in has no model client and performs no network calls.
        return None

    def short_memory_init(self) -> list[str]:
        return []

    def run(self, task: str) -> Any:
        self.short_memory.append(task)
        bidding = any(
            tool.get("function", {}).get("name") == "bid"
            for tool in self.tools_list_dictionary
        )
        if bidding:
            return {
                "function": {
                    "name": "bid",
                    "arguments": {
                        "confidence": self.confidence,
                        "estimated_cost": self.estimated_cost,
                    },
                }
            }

        self.executed_tasks.append(task)
        if self.fail_on_execute:
            raise RuntimeError(f"{self.agent_name}: simulated execution failure")
        return f"{self.agent_name} handled: {task}"


def make_pool() -> list[ScriptedAgent]:
    return [
        ScriptedAgent("Quality", confidence=0.95, estimated_cost=2.0),
        ScriptedAgent("Value", confidence=0.80, estimated_cost=0.5),
        ScriptedAgent("Generalist", confidence=0.60, estimated_cost=1.0),
    ]
