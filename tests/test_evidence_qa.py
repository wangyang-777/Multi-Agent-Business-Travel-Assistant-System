from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.agent.langgraph_orchestrator import LangGraphTravelOrchestrator
from app.core.rag.evidence import (
    AnswerDraft, AnswerReview, EvidenceError, EvidencePack, evaluate,
    numbers, render_draft, validate_draft, validate_pack, verification_result,
)
from app.domain.schemas import ChatMessage, MessageRole
from app.services.evidence_qa import EvidenceQA


class QueueLLM:
    model = "test"

    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = []

    async def chat_completion(self, messages, **kwargs):
        self.messages.append(messages)
        assert kwargs.get("response_format", {}).get("type") in {"json_object", "json_schema"}
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply, ensure_ascii=False)), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6, total_tokens=10),
        )


def allowance_pack(rate=120, days=3):
    return {
        "user_facts": [{"id": "q_days", "statement": f"出差{days}天", "quote": f"{days}个自然日", "value": str(days), "unit": "天"}],
        "policy_facts": [{"id": "p_rate", "statement": f"补助每天{rate}元", "sources": [{"chunk_id": "allowance", "quote": f"每天{rate} 元"}], "value": str(rate), "unit": "元/天"}],
        "requirements": [{"id": "r_total", "description": "补助合计", "status": "covered", "fact_ids": ["q_days", "p_rate"], "reason": "天数来自问题，日标准来自制度"}],
        "calculations": [{"id": "c_total", "expression": "q_days*p_rate", "unit": "元", "description": "天数乘日补助"}],
        "search_queries": [],
    }


def allowance_draft(amount=360):
    return {"claims": [{"id": "a_total", "text": f"伙食补助合计{amount}元。", "kind": "calculation", "fact_ids": ["c_total"], "requirement_ids": ["r_total"]}]}


def supported_review(claim_id="a_total"):
    return {"claims": [{"claim_id": claim_id, "status": "supported", "reason": "引用及计算均支持"}], "question_answered": True, "missing_information": []}


def planner(question, task_request=None):
    return {"primary_intent": "policy", "tasks": [{"id": "task_1", "intent": "policy", "request": task_request or question, "slots": {}, "depends_on": [], "missing_slots": []}], "clarification_question": None}


async def run_pipeline(monkeypatch, replies, question="出差3个自然日，未由其他单位供餐，伙食补助合计多少？", task_request=None, hits=None):
    llm = QueueLLM([planner(question, task_request), *replies])
    orch = LangGraphTravelOrchestrator(llm=llm)
    monkeypatch.setattr(orch, "_knowledge_retrieval_available", lambda: True)

    async def retrieve(query):
        return hits or [{"id": "allowance", "content": "每天120 元", "rerank_score": 0.9}]

    monkeypatch.setattr(orch, "_retrieve_knowledge", retrieve)
    result = await orch.run_completion([ChatMessage(role=MessageRole.USER, content=question)])
    return result, llm


@pytest.mark.parametrize("rate,days", [(120, 3), (140, 4), (95, 2)])
def test_different_policy_values_use_same_calculator(rate, days):
    citations = [{"chunk_id": "allowance", "content": f"出差每天{rate} 元"}]
    pack = EvidencePack.model_validate(allowance_pack(rate, days))
    records = validate_pack(pack, f"出差{days}个自然日", citations)
    draft = AnswerDraft.model_validate(allowance_draft(rate * days))
    validate_draft(draft, pack, records)
    result = verification_result(AnswerReview.model_validate(supported_review()), draft, records)
    assert result["passed"]
    assert f"{rate * days}元" in render_draft(draft, records, citations)
    assert result["claim_evidence_map"][0]["supporting_chunk_ids"] == ["allowance"]


@pytest.mark.parametrize("expression", ["__import__('os')", "x.real", "x**99999", "[x]", "x/0", "y+1", "1+2"])
def test_calculator_rejects_unsafe_or_unbound_expressions(expression):
    with pytest.raises(EvidenceError):
        evaluate(expression, {"x": Decimal(3)})


def test_calculator_handles_percent_and_boundary_comparisons():
    assert evaluate("base*(1+percent/100)", {"base": Decimal(500), "percent": Decimal(20)})[0] == "600"
    assert evaluate("15 < day <= 30", {"day": Decimal(20)})[0] is True
    assert evaluate("day < limit", {"day": Decimal(16), "limit": Decimal(16)})[0] is False


@pytest.mark.parametrize("change", ["quote", "source", "value", "user", "reference"])
def test_fabricated_quotes_values_and_ids_are_rejected(change):
    raw = allowance_pack()
    if change == "quote": raw["policy_facts"][0]["sources"][0]["quote"] = "每天999元"
    if change == "source": raw["policy_facts"][0]["sources"][0]["chunk_id"] = "invented"
    if change == "value": raw["policy_facts"][0]["value"] = "999"
    if change == "user": raw["user_facts"][0]["quote"] = "5个自然日"
    if change == "reference": raw["requirements"][0]["fact_ids"] = ["invented"]
    with pytest.raises(EvidenceError):
        validate_pack(EvidencePack.model_validate(raw), "3个自然日", [{"chunk_id": "allowance", "content": "每天120 元"}])


def test_wrong_result_cannot_pass_even_when_model_approves_it():
    pack = EvidencePack.model_validate(allowance_pack())
    records = validate_pack(pack, "3个自然日", [{"chunk_id": "allowance", "content": "每天120 元"}])
    result = verification_result(AnswerReview.model_validate(supported_review()), AnswerDraft.model_validate(allowance_draft(999)), records)
    assert not result["passed"]
    assert result["claim_evidence_map"][0]["status"] == "conflict"


def test_review_cannot_omit_claims():
    pack = EvidencePack.model_validate(allowance_pack())
    records = validate_pack(pack, "3个自然日", [{"chunk_id": "allowance", "content": "每天120 元"}])
    with pytest.raises(EvidenceError):
        verification_result(AnswerReview.model_validate(supported_review("another")), AnswerDraft.model_validate(allowance_draft()), records)


def test_chinese_user_number_and_whitespace_are_equivalent():
    raw = allowance_pack()
    raw["user_facts"][0]["quote"] = "三个自然日"
    records = validate_pack(EvidencePack.model_validate(raw), "出差三个自然日", [{"chunk_id": "allowance", "content": "每天 120\n元"}])
    assert records["c_total"]["result"] == "360"


async def test_correction_keeps_supported_claim_and_original_question():
    pack = EvidencePack.model_validate(allowance_pack())
    citations = [{"chunk_id": "allowance", "content": "每天120 元"}]
    question = "出差3个自然日，未由其他单位供餐"
    records = validate_pack(pack, question, citations)
    original = AnswerDraft.model_validate({"claims": [
        {"id": "a_rate", "text": "每天120元。", "kind": "policy", "fact_ids": ["p_rate"]},
        allowance_draft(999)["claims"][0],
    ]})
    llm = QueueLLM([allowance_draft()])
    corrected, _ = await EvidenceQA(llm).correct(question, pack, records, original, {
        "claim_evidence_map": [{"claim_id": "a_rate", "status": "supported"}, {"claim_id": "a_total", "status": "conflict"}],
    }, citations)
    payload = json.loads(llm.messages[0][1]["content"])
    assert payload["original_question"] == question
    assert corrected.claims[0] == original.claims[0]
    assert corrected.claims[1].text == "伙食补助合计360元。"


async def test_graph_preserves_conditions_and_exposes_all_stages(monkeypatch):
    result, _ = await run_pipeline(monkeypatch, [allowance_pack(), allowance_draft(), supported_review()], task_request="伙食补助合计多少？")
    assert "360元" in result["choices"][0]["message"]["content"]
    meta = result["metadata"]
    assert meta["verification"]["passed"] is True
    assert [s["stage"] for s in meta["rag_stages"]] == ["retrieval", "evidence", "draft", "verification", "final"]
    assert meta["rag_evidence"]["facts"]["q_days"]["value"] == "3"
    assert result["usage"]["total_tokens"] == 40


async def test_wrong_result_corrected_once_then_reverified(monkeypatch):
    result, _ = await run_pipeline(monkeypatch, [allowance_pack(), allowance_draft(999), supported_review(), allowance_draft(), supported_review()])
    meta = result["metadata"]
    assert meta["rag_correction_count"] == 1
    assert "360元" in result["choices"][0]["message"]["content"]
    assert "999" not in result["choices"][0]["message"]["content"]
    assert [s["stage"] for s in meta["rag_stages"]].count("verification") == 2
    assert result["usage"]["total_tokens"] == 60


async def test_verifier_failure_never_returns_unverified_draft(monkeypatch):
    result, _ = await run_pipeline(monkeypatch, [allowance_pack(), allowance_draft(999), RuntimeError("service unavailable")])
    assert "999" not in result["choices"][0]["message"]["content"]
    assert result["metadata"]["verification"]["passed"] is False


async def test_failed_correction_keeps_only_verified_claims(monkeypatch):
    draft = {"claims": [
        {"id": "a_rate", "text": "日补助标准120元。", "kind": "policy", "fact_ids": ["p_rate"]},
        allowance_draft(999)["claims"][0],
    ]}
    review = {"claims": [
        {"claim_id": "a_rate", "status": "supported", "reason": "原文支持"},
        {"claim_id": "a_total", "status": "contradicted", "reason": "计算错误"},
    ], "question_answered": False, "missing_information": []}
    result, _ = await run_pipeline(monkeypatch, [allowance_pack(), draft, review, RuntimeError("unavailable")])
    answer = result["choices"][0]["message"]["content"]
    assert "120元" in answer and "999" not in answer
    assert result["metadata"]["verification"]["question_answered"] is False
    assert result["metadata"]["verification"]["finalization"] == "supported_claims_only"


async def test_empty_structured_review_is_not_a_pass(monkeypatch):
    result, _ = await run_pipeline(monkeypatch, [allowance_pack(), allowance_draft(), {"claims": [], "question_answered": True}])
    assert result["metadata"]["verification"]["passed"] is False


async def test_missing_value_produces_scoped_abstention(monkeypatch):
    pack = {"requirements": [{"id": "r_ratio", "description": "早餐扣除比例", "status": "missing", "fact_ids": [], "reason": "只说按标准比例扣除"}], "search_queries": []}
    draft = {"claims": [{"id": "a_ratio", "text": "当前检索资料未提供早餐的具体扣除比例。", "kind": "limitation", "fact_ids": [], "requirement_ids": ["r_ratio"]}]}
    review = {"claims": [{"claim_id": "a_ratio", "status": "supported", "reason": "未发明扣除比例"}], "question_answered": False, "missing_information": ["具体比例"]}
    result, _ = await run_pipeline(monkeypatch, [pack, draft, review], question="早餐扣除百分之多少？", hits=[{"id": "ratio", "content": "供餐时按标准比例扣除。"}])
    assert "未提供" in result["choices"][0]["message"]["content"]
    assert result["metadata"]["verification"]["passed"] is True
    assert result["metadata"]["verification"]["question_answered"] is False


async def test_supplemental_retrieval_is_bounded_and_deduplicated(monkeypatch):
    pack = allowance_pack()
    missing = {**pack, "requirements": [{"id": "r_total", "description": "补助合计", "status": "missing", "fact_ids": ["q_days"], "reason": "需要补充标准"}], "search_queries": ["补助标准", "补助标准"]}
    llm = QueueLLM([missing, pack])
    orch = LangGraphTravelOrchestrator(llm=llm)
    calls = []

    async def retrieve(query):
        calls.append(query)
        return [{"id": "allowance", "content": "每天120 元"}, {"id": "extra", "content": "补充说明"}]

    monkeypatch.setattr(orch, "_retrieve_knowledge", retrieve)
    state = {"rag_question": "3个自然日", "citations": [{"chunk_id": "allowance", "content": "每天120 元"}], "trace": []}
    command = await orch._rag_evidence_builder(state)
    assert calls == ["补助标准"]
    assert len(command.update["citations"]) == 2


@pytest.mark.parametrize("grade,cabin", [("A职级", "公务舱"), ("总监", "经济舱"), ("P8", "高级经济舱")])
async def test_table_categories_are_document_driven(monkeypatch, grade, cabin):
    quote = f"| {grade} | {cabin} |"
    pack = {"policy_facts": [{"id": "p_cabin", "statement": f"{grade}飞机标准为{cabin}", "sources": [{"chunk_id": "table", "quote": quote}]}], "requirements": [{"id": "r_cabin", "description": "飞机标准", "status": "covered", "fact_ids": ["p_cabin"], "reason": "表格列明"}]}
    draft = {"claims": [{"id": "a_cabin", "text": f"{grade}飞机标准为{cabin}。", "kind": "policy", "fact_ids": ["p_cabin"]}]}
    result, _ = await run_pipeline(monkeypatch, [pack, draft, supported_review("a_cabin")], question=f"{grade}的飞机舱位是什么？", hits=[{"id": "table", "content": f"| 人员 | 飞机 |\n{quote}"}])
    assert cabin in result["choices"][0]["message"]["content"]
    assert result["metadata"]["verification"]["passed"] is True


async def test_invalid_quote_is_repaired_once_and_usage_includes_both_calls():
    invalid = allowance_pack()
    invalid["policy_facts"][0]["sources"][0]["quote"] = "每天999元"
    llm = QueueLLM([invalid, allowance_pack()])
    pack, records, usage = await EvidenceQA(llm).build(
        "3个自然日", [{"chunk_id": "allowance", "content": "每天120 元"}],
    )
    assert records["c_total"]["result"] == "360"
    assert usage["total_tokens"] == 20
    payload = json.loads(llm.messages[1][1]["content"])
    assert "invalid source quote" in payload["validation_error"]
    assert pack.policy_facts[0].value == "120"


async def test_second_invalid_quote_fails_instead_of_looping():
    invalid = allowance_pack()
    invalid["policy_facts"][0]["sources"][0]["quote"] = "每天999元"
    llm = QueueLLM([invalid, invalid])
    with pytest.raises(EvidenceError):
        await EvidenceQA(llm).build("3个自然日", [{"chunk_id": "allowance", "content": "每天120 元"}])
    assert len(llm.messages) == 2


async def test_schema_repair_is_bounded_and_cannot_approve_an_empty_review():
    invalid = {"claims": [], "question_answered": True}
    llm = QueueLLM([invalid, invalid])
    pack = EvidencePack.model_validate(allowance_pack())
    citations = [{"chunk_id": "allowance", "content": "每天120 元"}]
    records = validate_pack(pack, "3个自然日", citations)
    with pytest.raises(ValueError):
        await EvidenceQA(llm).review("3个自然日", pack, records, AnswerDraft.model_validate(allowance_draft()), citations)
    assert len(llm.messages) == 2


@pytest.mark.parametrize("quote", [" ", "\n\t"])
def test_blank_source_quote_cannot_validate_any_fact(quote):
    raw = allowance_pack()
    raw["policy_facts"][0]["sources"][0]["quote"] = quote
    with pytest.raises(ValueError):
        EvidencePack.model_validate(raw)


def test_calculation_cannot_invent_policy_rate_as_literal():
    raw = allowance_pack()
    raw["calculations"][0]["expression"] = "q_days*999"
    with pytest.raises(EvidenceError, match="constants"):
        validate_pack(EvidencePack.model_validate(raw), "3个自然日", [{"chunk_id": "allowance", "content": "每天120 元"}])


def test_digit_style_chinese_year_is_not_interpreted_as_last_digit():
    assert Decimal(2026) in numbers("二〇二六年")
    assert Decimal(120) in numbers("一百二十元")


async def test_request_timeout_cancels_model_work():
    import asyncio

    cancelled = asyncio.Event()

    class HangingLLM:
        async def chat_completion(self, *args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    with pytest.raises(TimeoutError):
        await EvidenceQA(HangingLLM(), timeout_seconds=0.01).build("问题", [])
    assert cancelled.is_set()


def test_partial_task_cannot_be_used_as_completed_prerequisite():
    result = LangGraphTravelOrchestrator._task_result(
        {"id": "p", "intent": "policy", "request": "差标及早餐比例"},
        {"answer": "差标120元，早餐比例未知", "answer_mode": "rag_grounded",
         "verification": {"passed": True, "question_answered": False}},
    )
    assert result["status"] == "needs_review"


async def test_multi_task_evidence_and_citations_stay_isolated(monkeypatch):
    from app.config import settings

    question = "出差3个自然日，伙食补助多少？P8飞机舱位是什么？"
    plan = {"primary_intent": "policy", "tasks": [
        {"id": "food", "intent": "policy", "request": "伙食补助多少？", "slots": {}, "depends_on": [], "missing_slots": []},
        {"id": "cabin", "intent": "policy", "request": "P8飞机舱位是什么？", "slots": {}, "depends_on": [], "missing_slots": []},
    ], "clarification_question": None}
    cabin_pack = {"policy_facts": [{"id": "p_cabin", "statement": "P8可乘公务舱", "sources": [{"chunk_id": "cabin", "quote": "P8 | 公务舱"}]}], "requirements": [{"id": "r_cabin", "description": "P8舱位", "status": "covered", "fact_ids": ["p_cabin"], "reason": "有表格依据"}]}
    cabin_draft = {"claims": [{"id": "a_cabin", "text": "P8可乘公务舱。", "kind": "policy", "fact_ids": ["p_cabin"]}]}
    llm = QueueLLM([plan, allowance_pack(), allowance_draft(), supported_review(), cabin_pack, cabin_draft, supported_review("a_cabin")])
    orch = LangGraphTravelOrchestrator(llm=llm)
    monkeypatch.setattr(settings, "task_max_concurrency", 1)
    monkeypatch.setattr(orch, "_knowledge_retrieval_available", lambda: True)

    async def retrieve(query):
        if "P8" in query:
            return [{"id": "cabin", "content": "| 人员 | 飞机 |\n| P8 | 公务舱 |"}]
        return [{"id": "allowance", "content": "每天120 元"}]

    monkeypatch.setattr(orch, "_retrieve_knowledge", retrieve)
    result = await orch.run_completion([ChatMessage(role=MessageRole.USER, content=question)])
    tasks = result["metadata"]["task_results"]
    assert [task["status"] for task in tasks] == ["completed", "completed"]
    assert tasks[0]["rag_evidence"]["facts"]["q_days"]["value"] == "3"
    assert "p_cabin" not in tasks[0]["rag_evidence"]["facts"]
    assert "p_rate" not in tasks[1]["rag_evidence"]["facts"]
    assert all(task["rag_stages"][-1]["stage"] == "final" for task in tasks)
    answer = result["choices"][0]["message"]["content"]
    assert "360元。[1]" in answer and "公务舱。[2]" in answer


@pytest.mark.parametrize("failure_stage", ["evidence", "verification"])
async def test_model_billing_error_is_distinguished_from_missing_policy(monkeypatch, failure_stage):
    class BillingError(Exception):
        status_code = 402

    error = BillingError("secret-provider-key-in-error-message")
    replies = [error] if failure_stage == "evidence" else [allowance_pack(), allowance_draft(999), error]
    result, _ = await run_pipeline(monkeypatch, replies)
    answer = result["choices"][0]["message"]["content"]
    assert "402" in answer and "账户" in answer
    assert "未检索到" not in answer and "999" not in answer
    assert "secret-provider-key" not in json.dumps(result)
    assert result["metadata"]["verification"]["upstream_status"] == 402


@pytest.mark.parametrize("change,code,field,record_id", [
    ("quote", "invalid_source_quote", "sources.0.quote", "p_rate"),
    ("source", "unknown_source", "sources.0.chunk_id", "p_rate"),
    ("value", "numeric_value_not_in_quote", "value", "p_rate"),
    ("user", "invalid_user_quote", "quote", "q_days"),
    ("reference", "unknown_requirement_fact", "fact_ids", "r_total"),
    ("formula", "unknown_numeric_fact", "expression", "c_total"),
])
def test_evidence_diagnostics_identify_failed_record_without_source_text(change, code, field, record_id):
    raw = allowance_pack()
    secret = "secret-key-and-source-text"
    if change == "quote": raw["policy_facts"][0]["sources"][0]["quote"] = secret
    if change == "source": raw["policy_facts"][0]["sources"][0]["chunk_id"] = secret
    if change == "value": raw["policy_facts"][0]["value"] = "999"
    if change == "user": raw["user_facts"][0]["quote"] = secret
    if change == "reference": raw["requirements"][0]["fact_ids"] = ["missing_fact"]
    if change == "formula": raw["calculations"][0]["expression"] = "q_days*missing_fact"
    with pytest.raises(EvidenceError) as caught:
        validate_pack(EvidencePack.model_validate(raw), "3个自然日", [{"chunk_id": "allowance", "content": "每天120 元"}])
    detail = caught.value.diagnostic()
    assert (detail["code"], detail["field"], detail["record_id"]) == (code, field, record_id)
    assert secret not in json.dumps(detail)
    if change in {"reference", "formula"}:
        assert detail["reference_ids"] == ["missing_fact"]


@pytest.mark.parametrize("error_type", ["fact", "requirement", "calculation", "limitation", "duplicate", "schema"])
async def test_draft_validation_repairs_once_then_runs_semantic_review(monkeypatch, error_type):
    invalid = allowance_draft()
    claim = invalid["claims"][0]
    if error_type == "fact": claim["fact_ids"] = ["invented_fact"]
    if error_type == "requirement": claim["requirement_ids"] = ["invented_requirement"]
    if error_type == "calculation": claim["fact_ids"] = ["p_rate"]
    if error_type == "limitation": claim.update(kind="limitation", requirement_ids=[])
    if error_type == "duplicate": invalid["claims"].append(dict(claim))
    if error_type == "schema": claim["kind"] = "unsupported-kind"
    result, llm = await run_pipeline(monkeypatch, [allowance_pack(), invalid, allowance_draft(), supported_review()])
    meta = result["metadata"]
    assert meta["verification"]["passed"] is True
    assert "360元" in result["choices"][0]["message"]["content"]
    stage = next(s for s in meta["rag_stages"] if s["stage"] == "draft")
    assert stage["repair_count"] == 1
    assert stage["diagnostics"][-1]["event"] == "repair_succeeded"
    assert result["usage"]["total_tokens"] == 50
    repair_input = json.loads(llm.messages[3][1]["content"])
    assert "previous_draft" in repair_input and "validation_detail" in repair_input


@pytest.mark.parametrize("first_schema_error,second_schema_error", [(False, False), (True, False), (False, True)])
async def test_schema_and_business_errors_share_one_draft_repair_budget(monkeypatch, first_schema_error, second_schema_error):
    invalid = allowance_draft()
    invalid["claims"][0]["fact_ids"] = ["invented_fact"]
    first = {"claims": []} if first_schema_error else invalid
    second = {"claims": []} if second_schema_error else invalid
    result, llm = await run_pipeline(monkeypatch, [allowance_pack(), first, second])
    assert len(llm.messages) == 4  # Planner, evidence, draft, one repair; no third draft call.
    verification = result["metadata"]["verification"]
    assert verification["passed"] is False and verification["stage"] == "draft"
    assert verification["repair_count"] == 1
    assert verification["diagnostics"][-1]["event"] == "repair_failed"
    assert verification["validation_error"]["code"] == (
        "schema_validation_failed" if second_schema_error else "unknown_claim_reference"
    )
    assert "360元" not in result["choices"][0]["message"]["content"]
    assert result["usage"]["total_tokens"] == 40


@pytest.mark.parametrize("days,threshold,rate", [(31, 30, 70), (16, 15, 85), (11, 10, 95)])
async def test_rule_selection_can_repair_misclassified_calculation_without_inventing_formula(monkeypatch, days, threshold, rate):
    quote = f"超过{threshold}天的，伙食补助每天{rate}元。"
    pack = {
        "user_facts": [{"id": "q_day", "statement": f"第{days}天", "quote": f"第{days}天", "value": str(days), "unit": "天"}],
        "policy_facts": [{"id": "p_rule", "statement": quote, "sources": [{"chunk_id": "rule", "quote": quote}], "value": str(rate), "unit": "元/天"}],
        "requirements": [{"id": "r_rate", "description": "对应天数标准", "status": "covered", "fact_ids": ["q_day", "p_rule"], "reason": "适用超过阈值的档位"}],
        "calculations": [],
    }
    valid = {"claims": [{"id": "a_rate", "text": f"第{days}天适用每天{rate}元的标准。", "kind": "policy", "fact_ids": ["q_day", "p_rule"], "requirement_ids": ["r_rate"]}]}
    invalid = json.loads(json.dumps(valid))
    invalid["claims"][0]["kind"] = "calculation"
    result, _ = await run_pipeline(monkeypatch, [pack, invalid, valid, supported_review("a_rate")], question=f"第{days}天伙食补助多少？", hits=[{"id": "rule", "content": quote}])
    assert result["metadata"]["verification"]["passed"] is True
    assert result["metadata"]["rag_evidence"]["pack"]["calculations"] == []
    draft_stage = next(s for s in result["metadata"]["rag_stages"] if s["stage"] == "draft")
    assert draft_stage["diagnostics"][0]["code"] == "calculation_without_checked_result"
    assert f"{rate}元" in result["choices"][0]["message"]["content"]


async def test_repaired_policy_kind_cannot_smuggle_new_amount_past_review(monkeypatch):
    invalid = allowance_draft(999)
    invalid["claims"][0]["fact_ids"] = ["p_rate"]
    wrong_repair = json.loads(json.dumps(invalid))
    wrong_repair["claims"][0]["kind"] = "policy"
    result, _ = await run_pipeline(monkeypatch, [allowance_pack(), invalid, wrong_repair, supported_review(), allowance_draft(), supported_review()])
    checks = [s for s in result["metadata"]["rag_stages"] if s["stage"] == "verification"]
    assert checks[0]["passed"] is False
    assert "999" not in result["choices"][0]["message"]["content"]
    assert "360元" in result["choices"][0]["message"]["content"]


async def test_failed_evidence_repair_keeps_specific_diagnostics_and_usage(monkeypatch):
    invalid = allowance_pack()
    invalid["policy_facts"][0]["sources"][0]["quote"] = "secret-text-not-in-source"
    result, _ = await run_pipeline(monkeypatch, [invalid, invalid])
    verification = result["metadata"]["verification"]
    assert verification["stage"] == "evidence"
    assert verification["validation_error"] == {"code": "invalid_source_quote", "field": "sources.0.quote", "record_id": "p_rate"}
    assert verification["repair_count"] == 1
    assert verification["diagnostics"][-1]["event"] == "repair_failed"
    assert "secret-text-not-in-source" not in json.dumps(result)
    assert result["usage"]["total_tokens"] == 30


async def test_schema_diagnostics_never_log_invalid_output_or_unknown_field_names(monkeypatch, caplog):
    secret = "sk-secret-provider-value"
    invalid = {"claims": [], secret: secret}
    result, _ = await run_pipeline(monkeypatch, [allowance_pack(), invalid, invalid])
    assert secret not in caplog.text
    assert secret not in json.dumps(result)
    fields = result["metadata"]["verification"]["validation_error"]["fields"]
    assert any(e["field"] == "claims" for e in fields)
    assert any(e["field"] == "<unknown_field>" for e in fields)


async def test_draft_repair_api_failure_does_not_return_invalid_draft(monkeypatch):
    class BillingError(Exception):
        status_code = 402

    invalid = allowance_draft()
    invalid["claims"][0]["fact_ids"] = ["invented_fact"]
    result, llm = await run_pipeline(monkeypatch, [allowance_pack(), invalid, BillingError("secret-key")])
    assert len(llm.messages) == 4
    assert "360元" not in result["choices"][0]["message"]["content"]
    assert "secret-key" not in json.dumps(result)
    assert result["metadata"]["verification"]["upstream_status"] == 402
    assert result["metadata"]["verification"]["diagnostics"][-1]["event"] == "repair_failed"


async def test_incomplete_draft_response_is_diagnosed_without_retry():
    class TruncatedLLM(QueueLLM):
        async def chat_completion(self, messages, **kwargs):
            response = await super().chat_completion(messages, **kwargs)
            response.choices[0].finish_reason = "length"
            return response

    citations = [{"chunk_id": "allowance", "content": "每天120 元"}]
    pack = EvidencePack.model_validate(allowance_pack())
    records = validate_pack(pack, "3个自然日", citations)
    llm = TruncatedLLM([allowance_draft()])
    diagnostics = []
    with pytest.raises(EvidenceError) as caught:
        await EvidenceQA(llm).draft("3个自然日", pack, records, citations, diagnostics=diagnostics)
    assert caught.value.code == "incomplete_model_response"
    assert len(llm.messages) == 1
    assert diagnostics == [{"operation": "answer_draft", "event": "validation_failed", "code": "incomplete_model_response", "field": "finish_reason"}]


async def test_correction_repairs_missing_requirement_once_and_is_still_reviewed():
    pack = EvidencePack.model_validate(allowance_pack())
    citations = [{"chunk_id": "allowance", "content": "每天120 元"}]
    records = validate_pack(pack, "3个自然日", citations)
    original = AnswerDraft.model_validate(allowance_draft())
    invalid = {"claims": [{"id": "a_total", "text": "需确认人员类别。", "kind": "limitation", "fact_ids": []}]}
    repaired = {"claims": [{**invalid["claims"][0], "requirement_ids": ["r_total"]}]}
    llm = QueueLLM([invalid, repaired])
    diagnostics = []
    corrected, usage = await EvidenceQA(llm).correct("3个自然日", pack, records, original, {"claim_evidence_map": [{"claim_id": "a_total", "status": "partial"}]}, citations, diagnostics=diagnostics)
    assert corrected.claims[0].requirement_ids == ["r_total"]
    assert usage["total_tokens"] == 20
    assert [e["event"] for e in diagnostics] == ["validation_failed", "repair_started", "repair_succeeded"]
    assert diagnostics[0]["code"] == "limitation_without_requirement"


async def test_uncertain_answer_creates_ticket_and_saves_only_pending_answer(monkeypatch):
    from app.services.human_review import HumanReviewStore
    from tests.test_chat_memory import _FakeRedis

    created = []
    async def create(self, session_id, prepared):
        created.append((session_id, prepared))
        assert "待人工审核" in str(await self.redis.get("chat:session:" + session_id))
        return {k: v for k, v in prepared.items() if k != "_snapshot"}
    monkeypatch.setattr(HumanReviewStore, "create", create)
    pack = {"requirements": [{"id": "r_ratio", "description": "早餐比例", "status": "missing", "fact_ids": [], "reason": "无具体比例"}], "search_queries": []}
    draft = {"claims": [{"id": "a_ratio", "text": "当前检索资料未提供早餐具体比例。", "kind": "limitation", "fact_ids": [], "requirement_ids": ["r_ratio"]}]}
    review = {"claims": [{"claim_id": "a_ratio", "status": "supported", "reason": "未编造"}], "question_answered": False, "missing_information": ["具体比例"]}
    question = "早餐扣减多少？"
    llm = QueueLLM([planner(question), pack, draft, review])
    redis = _FakeRedis()
    orch = LangGraphTravelOrchestrator(llm=llm, redis_client=redis)
    monkeypatch.setattr(orch, "_knowledge_retrieval_available", lambda: True)
    async def retrieve(query): return [{"id": "ratio", "content": "按标准比例扣除"}]
    monkeypatch.setattr(orch, "_retrieve_knowledge", retrieve)
    response = await orch.run_completion([ChatMessage(role=MessageRole.USER, content=question)], session_id="review-test")
    assert len(created) == 1
    assert "待人工审核" in response["choices"][0]["message"]["content"]
    assert response["metadata"]["human_reviews"][0]["status"] == "pending"
    assert "_snapshot" not in response["metadata"]["human_reviews"][0]
    assert created[0][1]["_snapshot"]["question"] == question


async def test_queue_unavailable_does_not_claim_successful_submission(monkeypatch):
    pack = {"requirements": [{"id": "r_ratio", "description": "早餐比例", "status": "missing", "fact_ids": [], "reason": "无具体比例"}], "search_queries": []}
    draft = {"claims": [{"id": "a_ratio", "text": "当前检索资料未提供早餐具体比例。", "kind": "limitation", "requirement_ids": ["r_ratio"]}]}
    review = {"claims": [{"claim_id": "a_ratio", "status": "supported", "reason": "未编造"}], "question_answered": False, "missing_information": ["具体比例"]}
    response, _ = await run_pipeline(monkeypatch, [pack, draft, review], question="早餐比例多少？")
    assert response["metadata"]["human_reviews"][0]["status"] == "submission_failed"
    assert response["metadata"]["human_reviews"][0]["review_id"] is None
    assert "提交失败" in response["choices"][0]["message"]["content"]
