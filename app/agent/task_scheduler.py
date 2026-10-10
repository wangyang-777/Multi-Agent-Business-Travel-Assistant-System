"""Bounded execution of validated task DAGs, with isolated results and partial failures."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

Task = dict[str, Any]
TaskResult = dict[str, Any]
TaskRunner = Callable[[Task, list[TaskResult]], Awaitable[TaskResult]]

# These intents are read-only; the executor also restricts their available tools.
PARALLEL_INTENTS = frozenset(
    {"search_flight", "search_hotel", "search_train", "policy", "rag", "info_query", "general"}
)


def unsuccessful_task(task: Task, status: str, answer: str) -> TaskResult:
    return {
        "task_id": task["id"],
        "intent": task["intent"],
        "request": task["request"],
        "status": status,
        "answer": answer,
        "risk_level": "medium",
        "citations": [],
        "tool_trace": [],
    }


async def execute_task_plan(
    tasks: list[Task],
    runner: TaskRunner,
    *,
    max_concurrency: int,
    timeout_seconds: float | Callable[[Task], float],
) -> list[TaskResult]:
    """Run only ready tasks; failed/unverified prerequisites block their descendants.

    Results retain plan order regardless of completion order. Non-read-only tasks
    run exclusively, so draft/approval work never races another task.
    """
    results: dict[str, TaskResult] = {}
    pending = {task["id"]: task for task in tasks}
    running: dict[asyncio.Task[TaskResult], Task] = {}

    async def execute(task: Task) -> TaskResult:
        dependencies = [results[task_id] for task_id in task["depends_on"]]
        try:
            timeout = timeout_seconds(task) if callable(timeout_seconds) else timeout_seconds
            return await asyncio.wait_for(runner(task, dependencies), timeout=timeout)
        except TimeoutError:
            return unsuccessful_task(task, "failed", "该任务执行超时，请稍后重试。")
        except Exception:  # Each task failure is isolated; cancellation still propagates.
            return unsuccessful_task(task, "failed", "该任务执行失败，请稍后重试。")

    try:
        while pending or running:
            # Iterate until all blocked descendants have been accounted for.
            blocked_any = False
            for task_id, task in list(pending.items()):
                failed = [
                    dep for dep in task["depends_on"]
                    if dep in results and results[dep]["status"] != "completed"
                ]
                if failed:
                    results[task_id] = unsuccessful_task(
                        task, "blocked", f"前置任务 {', '.join(failed)} 未可靠完成，该任务未执行。"
                    )
                    del pending[task_id]
                    blocked_any = True

            exclusive_running = any(
                task["intent"] not in PARALLEL_INTENTS for task in running.values()
            )
            if not exclusive_running:
                for task_id, task in list(pending.items()):
                    if len(running) >= max_concurrency:
                        break
                    if not all(dep in results for dep in task["depends_on"]):
                        continue
                    if any(results[dep]["status"] != "completed" for dep in task["depends_on"]):
                        continue
                    exclusive = task["intent"] not in PARALLEL_INTENTS
                    if exclusive and running:
                        continue
                    running[asyncio.create_task(execute(task))] = task
                    del pending[task_id]
                    if exclusive:
                        break

            if not running:
                if pending:
                    # May have just blocked an ancestor appearing later in plan order.
                    if blocked_any:
                        continue
                    raise ValueError("Task plan contains unresolved dependencies")
                break
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                task = running.pop(future)
                results[task["id"]] = future.result()
    finally:
        for future in running:
            future.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)

    return [results[task["id"]] for task in tasks]
