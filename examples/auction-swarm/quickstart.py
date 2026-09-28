"""Run deterministic auctions without a model API key."""

import os

os.environ["SWARMS_TELEMETRY_ON"] = "false"

from swarms.structs.auction_swarm import AuctionSwarm

from offline_agent import make_pool


def quality_first(confidence: float, estimated_cost: float) -> float:
    return confidence - 0.05 * estimated_cost


def main() -> None:
    task = "Summarize the refund policy."
    pool = make_pool()
    swarm = AuctionSwarm(
        name="cost-aware-auction",
        agents=pool,
        top_k=1,
        output_type="dict",
        print_on=False,
    )
    history = swarm.run(task)
    assert [message["role"] for message in history] == [
        "User", "cost-aware-auction", "Value"
    ]
    assert pool[0].executed_tasks == []
    assert pool[1].executed_tasks == [task]
    assert pool[2].executed_tasks == []
    assert [agent.short_memory for agent in pool] == [[], [task], []]
    assert all(agent.tools_list_dictionary == [] for agent in pool)
    print("Default winner:", history[-1]["role"])
    print("Answer:", history[-1]["content"])

    custom = AuctionSwarm(
        name="quality-first-auction",
        agents=make_pool(),
        top_k=2,
        scoring=quality_first,
        output_type="dict",
        print_on=False,
    )
    custom_history = custom.run(task)
    assert [message["role"] for message in custom_history] == [
        "User", "quality-first-auction", "Value", "Quality"
    ]
    print("Custom winner:", custom_history[-1]["role"])
    print("History roles:", [message["role"] for message in custom_history])


if __name__ == "__main__":
    main()
