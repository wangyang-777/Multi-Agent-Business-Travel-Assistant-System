import json

import pytest
from pydantic import ValidationError

from app.core.intent.recognizer import TravelIntent
from app.domain.task_plan import (
    ExecutionPlan,
    TaskSlots,
    execution_plan_json_schema,
    execution_plan_response_format,
)


def task(task_id="task_1", intent="policy", dependencies=None):
    return {
        "id": task_id,
        "intent": intent,
        "request": "查询差旅制度",
        "slots": TaskSlots().model_dump(mode="json"),
        "depends_on": dependencies or [],
        "missing_slots": [],
    }


def payload(tasks=None):
    return {
        "primary_intent": "policy",
        "tasks": tasks if tasks is not None else [task()],
        "clarification_question": None,
    }


def parse(data):
    return ExecutionPlan.model_validate_json(json.dumps(data, ensure_ascii=False))


def test_valid_plan_preserves_multiple_tasks_and_dependencies():
    data = payload([
        task("policy"),
        task("flight", "search_flight"),
        task("recommendation", "trip_planning", ["policy", "flight"]),
    ])
    data["tasks"][1]["slots"].update({
        "origin": "北京",
        "destination": "上海",
        "depart_date": "2026-10-01",
        "passengers": 2,
    })

    plan = parse(data)

    assert plan.primary_intent is TravelIntent.POLICY
    assert [item.id for item in plan.tasks] == ["policy", "flight", "recommendation"]
    assert plan.tasks[2].depends_on == ["policy", "flight"]
    assert plan.tasks[1].slots.non_null() == {
        "origin": "北京",
        "destination": "上海",
        "depart_date": "2026-10-01",
        "passengers": 2,
    }


@pytest.mark.parametrize("field", ["primary_intent", "tasks", "clarification_question"])
def test_missing_plan_fields_are_rejected(field):
    data = payload()
    del data[field]
    with pytest.raises(ValidationError):
        parse(data)


@pytest.mark.parametrize(
    "field", ["id", "intent", "request", "slots", "depends_on", "missing_slots"]
)
def test_missing_task_fields_are_rejected(field):
    data = payload()
    del data["tasks"][0][field]
    with pytest.raises(ValidationError):
        parse(data)


@pytest.mark.parametrize("value", ["unrecognized_intent", "", None, 1])
def test_unknown_intent_is_rejected_without_dropping_task(value):
    data = payload([task(), task("task_2", value)])
    with pytest.raises(ValidationError):
        parse(data)


@pytest.mark.parametrize("value", [[], [task(f"task_{i}") for i in range(9)]])
def test_empty_and_oversized_plans_are_rejected(value):
    with pytest.raises(ValidationError):
        parse(payload(value))


@pytest.mark.parametrize(
    ("tasks", "message"),
    [
        ([task(), task()], "unique"),
        ([task(dependencies=["unknown"])], "unknown dependencies"),
        ([task(dependencies=["task_1"])], "itself"),
        ([task(), task("task_2", dependencies=["task_1", "task_1"])], "duplicate"),
        (
            [task(dependencies=["task_2"]), task("task_2", dependencies=["task_1"])],
            "cycle",
        ),
    ],
)
def test_invalid_dependencies_are_rejected(tasks, message):
    with pytest.raises(ValidationError, match=message):
        parse(payload(tasks))


def test_primary_intent_must_belong_to_a_task():
    data = payload()
    data["primary_intent"] = "search_flight"
    with pytest.raises(ValidationError, match="primary_intent"):
        parse(data)


@pytest.mark.parametrize("field", ["id", "request"])
def test_blank_text_is_rejected(field):
    data = payload()
    data["tasks"][0][field] = "  \n "
    with pytest.raises(ValidationError):
        parse(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("passengers", "2"),
        ("passengers", True),
        ("passengers", 0),
        ("prefer_gd", "true"),
        ("origin", 1),
        ("origin", "   "),
        ("depart_date", "2026-02-30"),
        ("hotel_budget_cny", "500"),
        ("hotel_budget_cny", -1),
    ],
)
def test_slot_types_are_validated_without_silent_coercion(field, value):
    data = payload()
    data["tasks"][0]["slots"][field] = value
    with pytest.raises(ValidationError):
        parse(data)


@pytest.mark.parametrize("level", ["plan", "task", "slots"])
def test_unknown_properties_are_rejected(level):
    data = payload()
    target = {"plan": data, "task": data["tasks"][0], "slots": data["tasks"][0]["slots"]}
    target[level]["unexpected"] = "value"
    with pytest.raises(ValidationError, match="Extra inputs"):
        parse(data)


def test_schema_requires_nullable_slots_and_forbids_extra_keys():
    schema = execution_plan_json_schema()

    def verify(node):
        if isinstance(node, dict):
            assert "default" not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            if "$ref" in node:
                assert node["$ref"].startswith("#/$defs/")
            for value in node.values():
                verify(value)
        elif isinstance(node, list):
            for value in node:
                verify(value)

    verify(schema)
    slots_schema = schema["$defs"]["TaskSlots"]
    assert {"type": "null"} in slots_schema["properties"]["origin"]["anyOf"]
    response_format = execution_plan_response_format()
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == schema


def test_malformed_json_is_rejected():
    with pytest.raises(ValidationError):
        ExecutionPlan.model_validate_json('{"primary_intent":')
