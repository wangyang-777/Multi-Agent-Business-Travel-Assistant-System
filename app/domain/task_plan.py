"""Validated intent and task plan returned by the single planning model call."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from app.core.intent.recognizer import TravelIntent


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain non-whitespace characters")
    return value


NonEmptyText = Annotated[
    str,
    StringConstraints(min_length=1, strict=True),
    AfterValidator(_not_blank),
]
TaskId = Annotated[NonEmptyText, StringConstraints(max_length=64)]
PassengerCount = Annotated[int, Field(ge=1, le=9)]
MoneyAmount = Annotated[float, Field(ge=0)]


class _PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class TaskSlots(_PlanModel):
    """Known tool parameters; unset parameters are explicit nulls in model output.

    Defaults allow deterministic code to create an empty slot object. The exported
    response schema requires every property, including these nullable properties.
    Names follow the existing flight, hotel, train, and recommendation tools.
    """

    origin: NonEmptyText | None = None
    destination: NonEmptyText | None = None
    depart_date: date | None = None
    return_date: date | None = None
    cabin: NonEmptyText | None = None
    passengers: PassengerCount | None = None
    city: NonEmptyText | None = None
    check_in: date | None = None
    check_out: date | None = None
    keyword: NonEmptyText | None = None
    poi_name: NonEmptyText | None = None
    guests: PassengerCount | None = None
    max_nightly_cny: MoneyAmount | None = None
    origin_station: NonEmptyText | None = None
    dest_station: NonEmptyText | None = None
    prefer_gd: bool | None = None
    limited_num: Annotated[int, Field(ge=1, le=20)] | None = None
    employee_id: NonEmptyText | None = None
    grade: NonEmptyText | None = None
    origin_city: NonEmptyText | None = None
    destination_city: NonEmptyText | None = None
    departure_date: date | None = None
    purpose: NonEmptyText | None = None
    preferred_class: NonEmptyText | None = None
    hotel_budget_cny: MoneyAmount | None = None
    passenger_count: PassengerCount | None = None
    hotel_keyword: NonEmptyText | None = None
    hotel_nearby_poi: NonEmptyText | None = None
    include_trains: bool | None = None
    prefer_gd_trains: bool | None = None

    def non_null(self) -> dict[str, Any]:
        """Return only supplied parameters, with dates serialized as ISO strings."""
        return self.model_dump(mode="json", exclude_none=True)


class PlannedTask(_PlanModel):
    id: TaskId
    intent: TravelIntent
    request: NonEmptyText
    slots: TaskSlots
    depends_on: list[TaskId]
    missing_slots: list[NonEmptyText]

    @model_validator(mode="after")
    def validate_dependency_list(self) -> Self:
        if self.id in self.depends_on:
            raise ValueError(f"task {self.id!r} cannot depend on itself")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError(f"task {self.id!r} has duplicate dependencies")
        return self


class ExecutionPlan(_PlanModel):
    primary_intent: TravelIntent
    tasks: list[PlannedTask] = Field(min_length=1, max_length=8)
    clarification_question: NonEmptyText | None

    @model_validator(mode="after")
    def validate_task_graph(self) -> Self:
        tasks_by_id = {task.id: task for task in self.tasks}
        if len(tasks_by_id) != len(self.tasks):
            raise ValueError("task IDs must be unique")
        if not any(task.intent == self.primary_intent for task in self.tasks):
            raise ValueError("primary_intent must be present among the task intents")
        for task in self.tasks:
            unknown = set(task.depends_on) - tasks_by_id.keys()
            if unknown:
                raise ValueError(f"task {task.id!r} has unknown dependencies: {sorted(unknown)}")

        visiting: set[str] = set()
        completed: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("task dependencies contain a cycle")
            if task_id in completed:
                return
            visiting.add(task_id)
            for dependency in tasks_by_id[task_id].depends_on:
                visit(dependency)
            visiting.remove(task_id)
            completed.add(task_id)

        for task_id in tasks_by_id:
            visit(task_id)
        return self


def execution_plan_json_schema() -> dict[str, Any]:
    """Return a closed JSON Schema with all object properties required.

    Nullable slots remain optional in meaning, but their keys must be present in
    structured model output. Local ``$defs`` references are retained.
    """
    schema = ExecutionPlan.model_json_schema()

    def require_properties(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                require_properties(value)
        elif isinstance(node, list):
            for value in node:
                require_properties(value)

    require_properties(schema)
    return schema


def execution_plan_response_format() -> dict[str, Any]:
    """Response format for a client supporting strict JSON Schema output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "execution_plan",
            "strict": True,
            "schema": execution_plan_json_schema(),
        },
    }
