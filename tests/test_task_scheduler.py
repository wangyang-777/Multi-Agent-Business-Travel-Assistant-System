from __future__ import annotations

import asyncio

import pytest

from app.agent.task_scheduler import execute_task_plan


def task(task_id, intent="policy", dependencies=()):
    return {
        "id": task_id,
        "intent": intent,
        "request": f"request for {task_id}",
        "depends_on": list(dependencies),
    }


def result(task_id, status="completed"):
    return {"task_id": task_id, "status": status, "answer": f"answer for {task_id}"}


async def observed(event):
    # A bounded wait makes a scheduler regression fail instead of hanging the suite.
    await asyncio.wait_for(event.wait(), timeout=1)


async def stop_future(future):
    if not future.done():
        future.cancel()
    await asyncio.gather(future, return_exceptions=True)


async def test_independent_read_only_tasks_overlap_with_bounded_concurrency():
    tasks = [
        task("flight", "search_flight"),
        task("hotel", "search_hotel"),
        task("train", "search_train"),
        task("policy"),
    ]
    started = {item["id"]: asyncio.Event() for item in tasks}
    release = {item["id"]: asyncio.Event() for item in tasks}
    active = 0
    peak = 0

    async def runner(item, dependencies):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        started[item["id"]].set()
        try:
            await release[item["id"]].wait()
            return result(item["id"])
        finally:
            active -= 1

    future = asyncio.create_task(
        execute_task_plan(tasks, runner, max_concurrency=2, timeout_seconds=2)
    )
    try:
        await observed(started["flight"])
        await observed(started["hotel"])
        assert active == 2
        assert not started["train"].is_set()
        assert not started["policy"].is_set()

        release["flight"].set()
        await observed(started["train"])
        assert active == 2
        assert not started["policy"].is_set()

        release["hotel"].set()
        await observed(started["policy"])
        assert active == 2
        release["train"].set()
        release["policy"].set()
        results = await future
    finally:
        await stop_future(future)

    assert peak == 2
    assert active == 0
    assert [item["task_id"] for item in results] == ["flight", "hotel", "train", "policy"]
    assert all(item["status"] == "completed" for item in results)


async def test_dependencies_wait_and_results_preserve_reverse_listed_plan_order():
    tasks = [task("child", dependencies=["parent"]), task("parent"), task("sibling")]
    parent_started = asyncio.Event()
    parent_release = asyncio.Event()
    child_started = asyncio.Event()
    sibling_done = asyncio.Event()
    dependencies_seen = []

    async def runner(item, dependencies):
        if item["id"] == "parent":
            parent_started.set()
            await parent_release.wait()
        elif item["id"] == "child":
            dependencies_seen.extend(dependencies)
            child_started.set()
        else:
            sibling_done.set()
        return result(item["id"])

    future = asyncio.create_task(
        execute_task_plan(tasks, runner, max_concurrency=3, timeout_seconds=2)
    )
    try:
        await observed(parent_started)
        await observed(sibling_done)
        assert not child_started.is_set()
        parent_release.set()
        results = await future
    finally:
        await stop_future(future)

    assert child_started.is_set()
    assert dependencies_seen == [result("parent")]
    assert [item["task_id"] for item in results] == ["child", "parent", "sibling"]


@pytest.mark.parametrize("status", ["failed", "needs_review"])
async def test_unsuccessful_dependencies_block_reverse_listed_transitive_descendants(status):
    tasks = [
        task("grandchild", dependencies=["child"]),
        task("child", dependencies=["parent"]),
        task("parent"),
        task("sibling"),
    ]
    executed = []

    async def runner(item, dependencies):
        executed.append(item["id"])
        return result(item["id"], status if item["id"] == "parent" else "completed")

    results = await execute_task_plan(tasks, runner, max_concurrency=2, timeout_seconds=1)

    assert set(executed) == {"parent", "sibling"}
    assert [item["status"] for item in results] == ["blocked", "blocked", status, "completed"]
    assert "child" in results[0]["answer"]
    assert "parent" in results[1]["answer"]


async def test_exception_is_isolated_and_successful_sibling_result_survives():
    async def runner(item, dependencies):
        if item["id"] == "broken":
            raise RuntimeError("private implementation details")
        return result(item["id"])

    results = await execute_task_plan(
        [task("broken"), task("sibling"), task("dependent", dependencies=["broken"])],
        runner,
        max_concurrency=2,
        timeout_seconds=1,
    )

    assert [item["status"] for item in results] == ["failed", "completed", "blocked"]
    assert "private implementation details" not in results[0]["answer"]
    assert results[1] == result("sibling")


async def test_timeout_cancels_timed_out_runner_and_sibling_survives():
    cancelled = asyncio.Event()
    never_released = asyncio.Event()

    async def runner(item, dependencies):
        if item["id"] == "slow":
            try:
                await never_released.wait()
            finally:
                cancelled.set()
        return result(item["id"])

    results = await execute_task_plan(
        [task("slow"), task("sibling"), task("dependent", dependencies=["slow"])],
        runner,
        max_concurrency=2,
        timeout_seconds=0.01,
    )

    assert cancelled.is_set()
    assert [item["status"] for item in results] == ["failed", "completed", "blocked"]
    assert "超时" in results[0]["answer"]
    assert results[1] == result("sibling")


@pytest.mark.parametrize("intent", ["booking", "application", "trip_planning"])
async def test_non_parallel_task_runs_exclusively(intent):
    tasks = [task("before"), task("exclusive", intent), task("after")]
    started = {item["id"]: asyncio.Event() for item in tasks}
    release = {item["id"]: asyncio.Event() for item in tasks}
    active = set()
    exclusive_overlap = []

    async def runner(item, dependencies):
        if item["id"] == "exclusive":
            exclusive_overlap.extend(active)
        elif "exclusive" in active:
            exclusive_overlap.append(item["id"])
        active.add(item["id"])
        started[item["id"]].set()
        try:
            await release[item["id"]].wait()
            return result(item["id"])
        finally:
            active.remove(item["id"])

    future = asyncio.create_task(
        execute_task_plan(tasks, runner, max_concurrency=3, timeout_seconds=2)
    )
    try:
        await observed(started["before"])
        await observed(started["after"])
        assert not started["exclusive"].is_set()
        release["before"].set()
        release["after"].set()
        await observed(started["exclusive"])
        assert active == {"exclusive"}
        release["exclusive"].set()
        results = await future
    finally:
        await stop_future(future)

    assert not exclusive_overlap
    assert not active
    assert [item["task_id"] for item in results] == ["before", "exclusive", "after"]
    assert all(item["status"] == "completed" for item in results)


async def test_exclusive_first_task_prevents_later_read_only_tasks_starting():
    exclusive_started = asyncio.Event()
    exclusive_release = asyncio.Event()
    read_started = asyncio.Event()

    async def runner(item, dependencies):
        if item["id"] == "exclusive":
            exclusive_started.set()
            await exclusive_release.wait()
        else:
            read_started.set()
        return result(item["id"])

    future = asyncio.create_task(
        execute_task_plan(
            [task("exclusive", "booking"), task("read")],
            runner,
            max_concurrency=3,
            timeout_seconds=2,
        )
    )
    try:
        await observed(exclusive_started)
        assert not read_started.is_set()
        exclusive_release.set()
        results = await future
    finally:
        await stop_future(future)

    assert read_started.is_set()
    assert all(item["status"] == "completed" for item in results)


async def test_cancelling_scheduler_awaits_all_running_runners_cleanup():
    tasks = [task("a"), task("b"), task("pending")]
    started = {item["id"]: asyncio.Event() for item in tasks}
    cancelled = {item["id"]: asyncio.Event() for item in tasks}
    never_released = asyncio.Event()
    runner_futures = []

    async def runner(item, dependencies):
        runner_futures.append(asyncio.current_task())
        started[item["id"]].set()
        try:
            await never_released.wait()
        finally:
            cancelled[item["id"]].set()
        return result(item["id"])

    future = asyncio.create_task(
        execute_task_plan(tasks, runner, max_concurrency=2, timeout_seconds=2)
    )
    try:
        await observed(started["a"])
        await observed(started["b"])
        future.cancel()
        with pytest.raises(asyncio.CancelledError):
            await future
    finally:
        await stop_future(future)

    assert cancelled["a"].is_set()
    assert cancelled["b"].is_set()
    assert not started["pending"].is_set()
    assert all(item.done() for item in runner_futures)


async def test_partial_invalid_dependency_graph_raises_instead_of_spinning():
    async def runner(item, dependencies):
        return result(item["id"])

    with pytest.raises(ValueError, match="unresolved dependencies"):
        await execute_task_plan(
            [task("completed"), task("stuck", dependencies=["completed", "absent"])],
            runner,
            max_concurrency=2,
            timeout_seconds=1,
        )


async def test_task_specific_timeout_preserves_other_task_result():
    timeouts_seen = []

    def timeout_for(item):
        timeouts_seen.append(item["id"])
        return 0.01 if item["id"] == "slow" else 1.0

    async def runner(item, dependencies):
        if item["id"] == "slow":
            await asyncio.Event().wait()
        return result(item["id"])

    results = await execute_task_plan(
        [task("slow"), task("fast", "search_flight")], runner,
        max_concurrency=2, timeout_seconds=timeout_for,
    )
    assert set(timeouts_seen) == {"slow", "fast"}
    assert [item["status"] for item in results] == ["failed", "completed"]
