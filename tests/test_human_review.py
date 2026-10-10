from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import human_reviews
from app.config import settings
from app.services.human_review import prepare_review, public_result, review_reasons


def uncertain_state():
    return {
        "rag_question": "某人员类别的补贴合计多少？", "rag_task_request": "计算补贴",
        "rag_stages": [{"stage": "verification"}],
        "rag_evidence": {"pack": {"requirements": [{"id": "r_grade", "status": "missing", "description": "人员类别", "reason": "对应关系未规定"}]}},
        "verification": {"passed": True, "question_answered": False, "missing_information": ["人员类别"]},
        "citations": [{"chunk_id": "p1", "content": "某类人员每人每天100元。"}],
        "rag_draft": {"claims": []},
    }


def test_supported_claims_with_missing_conditions_still_require_a_human():
    state = uncertain_state()
    review = prepare_review(state, "候选100元")
    assert review["status"] == "pending"
    assert review["_snapshot"]["candidate_answer"] == "候选100元"
    assert review["_snapshot"]["citations"][0]["content"]
    assert {reason["code"] for reason in review["reasons"]} == {"evidence_missing", "missing_information"}


def test_certain_answer_and_unrelated_general_chat_do_not_create_tickets():
    assert review_reasons({"answer": "你好"}) == []
    state = {"rag_stages": [{"stage": "verification"}], "verification": {"passed": True, "question_answered": True}}
    assert prepare_review(state, "明确答案") is None


@pytest.mark.parametrize("reason", ["correction_failed:EvidenceError", "correction_exhausted", "verification_unavailable:TimeoutError"])
def test_failed_or_exhausted_processing_cannot_be_treated_as_finished(reason):
    state = {"rag_stages": [{"stage": "verification"}], "verification": {"passed": True, "question_answered": False, "reason": reason}}
    assert any(r["code"] == "processing_incomplete" for r in review_reasons(state))


def test_explicit_semantic_disagreement_can_refer_even_a_supported_answer():
    state = {"rag_stages": [{"stage": "verification"}], "verification": {"passed": True, "question_answered": True, "human_review_required": True, "review_reasons": ["版本适用不确定"]}}
    assert review_reasons(state) == [{"code": "semantic_uncertainty", "detail": "版本适用不确定"}]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "human_review_api_token", "test-operator-token")
    monkeypatch.setattr(settings, "human_review_reviewer_name", "财务审核账号")
    repo = SimpleNamespace(get=AsyncMock(), pending=AsyncMock(return_value=[]), decide=AsyncMock(), for_session=AsyncMock(return_value=[]))
    app = FastAPI()
    app.include_router(human_reviews.router, prefix="/api/v1")
    app.dependency_overrides[human_reviews.store] = lambda: repo
    with TestClient(app) as c:
        yield c, repo


def ticket(status="pending"):
    return {"review_id": str(uuid4()), "session_id": "s1", "result_token": "owner-receipt-token", "status": status, "reasons": [{"code": "missing", "detail": "类别未确认"}], "created_at": "2026-10-08", "expires_at": "2026-11-08", "decision": None, "snapshot": {"question": "Q", "task_request": "Q", "candidate_answer": "未经审核的999元", "citations": [], "evidence": {"private": "review snapshot"}}}


def headers():
    return {"Authorization": "Bearer test-operator-token"}


@pytest.mark.parametrize("path", ["/api/v1/human-reviews", "/api/v1/human-reviews/" + str(uuid4())])
def test_queue_and_detail_require_operator_credential(client, path):
    c, repo = client
    assert c.get(path).status_code == 401
    assert c.get(path, headers={"Authorization": "Bearer invalid"}).status_code == 401
    repo.get.assert_not_called()
    repo.pending.assert_not_called()


def test_missing_server_credential_fails_closed(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(settings, "human_review_api_token", "")
    assert c.get("/api/v1/human-reviews", headers=headers()).status_code == 503


def test_pending_user_receipt_does_not_expose_candidate_or_private_snapshot(client):
    c, repo = client
    t = ticket(); repo.get.return_value = t
    result = c.get(f'/api/v1/human-reviews/{t["review_id"]}/result', params={"session_id": "s1"}, headers={"X-Review-Result-Token": "owner-receipt-token"})
    assert result.status_code == 200
    assert result.json()["final_answer"] == ""
    assert "999" not in result.text and "private" not in result.text
    assert c.get(f'/api/v1/human-reviews/{t["review_id"]}/result', params={"session_id": "s1"}).status_code == 404
    assert c.get(f'/api/v1/human-reviews/{t["review_id"]}/result', params={"session_id": "s1"}, headers={"X-Review-Result-Token": "wrong-token"}).status_code == 404
    assert c.get(f'/api/v1/human-reviews/{t["review_id"]}/result', params={"session_id": "someone-else"}, headers={"X-Review-Result-Token": "owner-receipt-token"}).status_code == 404


@pytest.mark.parametrize("body", [
    {"action": "approved", "notes": "已核对", "final_answer": " "},
    {"action": "approved", "notes": " ", "final_answer": "100元"},
    {"action": "rejected", "notes": "条件不符", "final_answer": "100元"},
])
def test_incomplete_or_nonapproved_final_answers_are_rejected(client, body):
    c, repo = client
    assert c.post(f'/api/v1/human-reviews/{uuid4()}/decision', headers=headers(), json=body).status_code == 422
    repo.decide.assert_not_called()


def test_decision_endpoint_cannot_be_called_by_chat_user(client):
    c, repo = client
    assert c.post(f'/api/v1/human-reviews/{uuid4()}/decision', json={"action": "approved", "final_answer": "100元", "notes": "是"}).status_code == 401
    repo.decide.assert_not_called()


def test_reviewer_identity_comes_from_server_and_final_answer_is_not_rewritten(client):
    c, repo = client
    t = ticket("approved"); t["decision"] = {"final_answer": "经确认适用A类别，上限100元。", "notes": "已核对适用范围", "reviewer": "财务审核账号", "reviewed_at": "now"}; repo.decide.return_value = t
    result = c.post(f'/api/v1/human-reviews/{t["review_id"]}/decision', headers=headers(), json={"action": "approved", "final_answer": t["decision"]["final_answer"], "notes": "已核对适用范围", "reviewer": "伪造审核员"})
    assert result.status_code == 200
    assert result.json()["final_answer"] == t["decision"]["final_answer"]
    assert repo.decide.await_args.kwargs["reviewer"] == "财务审核账号"


def test_existing_decision_cannot_be_silently_replaced(client):
    c, repo = client
    repo.decide.side_effect = human_reviews.ReviewConflict("已经审核")
    assert c.post(f'/api/v1/human-reviews/{uuid4()}/decision', headers=headers(), json={"action": "approved", "final_answer": "100元", "notes": "确认"}).status_code == 409


def test_rejected_receipt_never_publishes_final_amount():
    t = ticket("needs_information"); t["decision"] = {"notes": "请确认职级", "reviewer": "财务", "final_answer": ""}
    result = public_result(t)
    assert result["final_answer"] == "" and result["citations"] == []


async def test_stream_done_includes_review_status_and_receipt_capability(monkeypatch):
    from app.agent.langgraph_orchestrator import LangGraphTravelOrchestrator
    from app.domain.schemas import ChatMessage, MessageRole
    orch = LangGraphTravelOrchestrator(llm=SimpleNamespace(model="test"))
    monkeypatch.setattr(orch, "run_completion", AsyncMock(return_value={
        "choices": [{"message": {"content": "待人工审核"}}], "session_id": "s1",
        "metadata": {"answer_status": "pending_human_review", "human_reviews": [{"review_id": "r1", "result_token": "receipt"}]},
    }))
    chunks = [chunk async for chunk in orch.stream_completion([ChatMessage(role=MessageRole.USER, content="Q")])]
    assert ''.join(chunk.delta or '' for chunk in chunks) == "待人工审核"
    assert chunks[-1].session_id == "s1"
    assert chunks[-1].human_reviews[0]["result_token"] == "receipt"
    assert chunks[-1].answer_status == "pending_human_review"


async def test_multi_task_tickets_are_persisted_once_and_snapshots_stay_private(monkeypatch):
    from app.agent.langgraph_orchestrator import LangGraphTravelOrchestrator
    from app.domain.schemas import ChatMessage, MessageRole
    from app.services.human_review import HumanReviewStore
    from tests.test_chat_memory import _FakeRedis
    redis = _FakeRedis()
    orch = LangGraphTravelOrchestrator(llm=SimpleNamespace(model="test"), redis_client=redis)
    calls = []
    async def create(self, session_id, prepared):
        calls.append(prepared["task_id"])
        return {k: v for k, v in prepared.items() if k != '_snapshot'}
    monkeypatch.setattr(HumanReviewStore, 'create', create)
    messages = [ChatMessage(role=MessageRole.USER, content="审核两个问题")]
    tasks = [{"task_id": task_id, "answer": "待人工审核", "human_review": {"review_id": str(uuid4()), "status": "pending", "reasons": [], "task_id": task_id, "_snapshot": {"private": "raw"}}} for task_id in ('a', 'b')]
    state = {"messages": messages, "conversation_messages": messages, "session_id": "multi-review", "route": "multi_task", "answer": "两个问题待人工审核", "task_results": tasks}
    response = (await orch._response_finalizer(state))["response"]
    assert calls == ['a', 'b']
    assert len(response["metadata"]["human_reviews"]) == 2
    assert response["metadata"]["answer_status"] == 'pending_human_review'
    assert '_snapshot' not in str(response["metadata"]["task_results"])


async def test_timed_out_policy_task_is_referred_instead_of_published_as_automatic(monkeypatch):
    import app.agent.langgraph_orchestrator as graph
    from app.domain.schemas import ChatMessage, MessageRole
    orch = graph.LangGraphTravelOrchestrator(llm=SimpleNamespace(model="test"))
    monkeypatch.setattr(graph, 'execute_task_plan', AsyncMock(return_value=[{
        'task_id': 'policy-task', 'intent': 'policy', 'request': '补助标准', 'status': 'failed', 'answer': '任务超时', 'citations': [], 'tool_trace': [],
    }]))
    command = await orch._multi_task_executor({'messages': [ChatMessage(role=MessageRole.USER, content='补助标准和其他任务')], 'execution_plan': {'tasks': []}})
    task = command.update['task_results'][0]
    assert task['status'] == 'needs_review'
    assert task['human_review']['_snapshot']['task_request'] == '补助标准'
    assert '待人工审核' in task['answer']
