from __future__ import annotations

from app.agent.langgraph_orchestrator import LangGraphTravelOrchestrator
from app.core.intent.recognizer import TravelIntent


def test_inventory_planning_request_routes_through_policy_reasoner() -> None:
    orch = LangGraphTravelOrchestrator()

    route = orch._route_after_intent(
        TravelIntent.TRIP_PLANNING.value,
        "我是staff，下周一从北京到上海出差一天，请综合推荐航班、高铁和酒店，并做差标提醒",
    )

    assert route == "policy_reasoner"


def test_pure_policy_question_still_uses_rag() -> None:
    orch = LangGraphTravelOrchestrator()

    route = orch._route_after_intent(
        TravelIntent.POLICY.value,
        "staff 去上海酒店标准是多少",
    )

    assert route == "rag_responder"
