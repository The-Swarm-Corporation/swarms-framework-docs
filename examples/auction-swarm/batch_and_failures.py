"""Exercise sequential batches, execution failures, and zero-score ties."""

import os

os.environ["SWARMS_TELEMETRY_ON"] = "false"

from swarms.structs.auction_swarm import AuctionSwarm

from offline_agent import ScriptedAgent


def main() -> None:
    worker = ScriptedAgent("Worker", confidence=0.9, estimated_cost=1.0)
    batch = AuctionSwarm(
        name="batch-auction",
        agents=[worker],
        output_type="dict",
        print_on=False,
    )
    tasks = ["Write release notes.", "Summarize a bug report."]
    histories = batch.batch_run(tasks)
    assert worker.executed_tasks == tasks
    assert [history[0]["content"] for history in histories] == tasks
    assert all(len(history) == 3 for history in histories)
    print("Batch answers:", [history[-1]["content"] for history in histories])

    primary = ScriptedAgent("Primary", 0.95, 1.0, fail_on_execute=True)
    backup = ScriptedAgent("Backup", 0.8, 1.0)
    unselected = ScriptedAgent("Unselected", 0.7, 1.0)
    fallback = AuctionSwarm(
        name="fallback-auction",
        agents=[primary, backup, unselected],
        top_k=2,
        output_type="dict",
        print_on=False,
    )
    history = fallback.run("Prepare a support response.")
    assert [message["role"] for message in history] == [
        "User", "fallback-auction", "Backup"
    ]
    assert unselected.executed_tasks == []
    print("Fallback winner:", history[-1]["role"])

    all_failed = AuctionSwarm(
        name="failed-auction",
        agents=[ScriptedAgent("Unavailable", 0.9, 1.0, fail_on_execute=True)],
        output_type="dict",
        print_on=False,
    ).run("Prepare a support response.")
    assert [message["role"] for message in all_failed] == ["User", "failed-auction"]
    print("All failed: no execution response")

    zero_pool = [
        ScriptedAgent("First", 0.0, 1.0),
        ScriptedAgent("Second", 0.0, 1.0),
        ScriptedAgent("Third", 0.0, 1.0),
    ]
    zero = AuctionSwarm(
        name="zero-auction",
        agents=zero_pool,
        top_k=2,
        output_type="dict",
        print_on=False,
    ).run("Handle an unfamiliar task.")
    assert [message["role"] for message in zero] == [
        "User", "zero-auction", "Second", "First"
    ]
    assert zero_pool[2].executed_tasks == []
    print("Zero-score history roles:", [message["role"] for message in zero])


if __name__ == "__main__":
    main()
