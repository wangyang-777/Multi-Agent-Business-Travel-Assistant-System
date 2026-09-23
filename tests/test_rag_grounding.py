from __future__ import annotations

from app.agent.langgraph_orchestrator import LangGraphTravelOrchestrator


def test_grounding_builds_supported_claim_evidence_map() -> None:
    result = LangGraphTravelOrchestrator._verify_answer_grounding(
        "staff 员工在上海住宿，酒店标准为 800 CNY。",
        [
            {
                "chunk_id": "chunk-hotel-1",
                "content": "staff 员工在上海住宿时，酒店标准为每晚不超过 800 CNY。",
            }
        ],
        "rag_grounded",
    )

    assert result["passed"] is True
    assert result["claim_evidence_map"][0]["status"] == "supported"
    assert result["claim_evidence_map"][0]["supporting_chunk_ids"] == ["chunk-hotel-1"]


def test_grounding_marks_conflicting_amount() -> None:
    result = LangGraphTravelOrchestrator._verify_answer_grounding(
        "staff 员工在上海的酒店标准为 900 CNY。",
        [
            {
                "chunk_id": "chunk-hotel-1",
                "content": "staff 员工在上海的酒店标准为 800 CNY。",
            }
        ],
        "rag_grounded",
    )

    assert result["passed"] is False
    assert result["claim_evidence_map"][0]["status"] == "conflict"
    assert "900 CNY" in result["unsupported_terms"]


async def test_grounding_runs_one_correction_then_forces_safe_fallback() -> None:
    orchestrator = LangGraphTravelOrchestrator()
    state = {
        "answer": "staff 员工在上海的酒店标准为 900 CNY。",
        "citations": [
            {
                "chunk_id": "chunk-hotel-1",
                "content": "staff 员工在上海的酒店标准为 800 CNY。",
            }
        ],
        "answer_mode": "rag_grounded",
        "risk_level": "low",
        "rag_correction_count": 0,
        "trace": [],
    }

    first = await orchestrator._grounding_verifier(state)  # type: ignore[arg-type]
    assert first.goto == "rag_self_corrector"

    second_state = {**state, **first.update, "rag_correction_count": 1}
    second = await orchestrator._grounding_verifier(second_state)  # type: ignore[arg-type]
    assert second.goto == "response_finalizer"
    assert second.update["answer_mode"] == "llm_fallback"
    assert "无法可靠给出公司制度结论" in second.update["answer"]
