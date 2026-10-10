"""Persist evidence review requests and audit operator decisions in Redis."""
from __future__ import annotations

import json
import secrets
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from redis.exceptions import WatchError

from app.config import settings

INDEX = "human_review:pending"


def review_key(review_id: str) -> str:
    return f"human_review:ticket:{review_id}"


def review_reasons(state: dict[str, Any]) -> list[dict[str, str]]:
    """Use observable evidence gaps and failures, not a model confidence percentage."""
    if not state.get("rag_stages") and not state.get("rag_evidence") and not (state.get("verification") or {}).get("stage"):
        return []
    verification = state.get("verification") or {}
    reasons = []
    for requirement in ((state.get("rag_evidence") or {}).get("pack") or {}).get("requirements", []):
        if requirement.get("status") in {"missing", "conflict"}:
            reasons.append({"code": "evidence_" + requirement["status"], "detail": requirement["description"] + "：" + requirement.get("reason", "")})
    for detail in verification.get("missing_information") or []:
        reasons.append({"code": "missing_information", "detail": str(detail)})
    if verification.get("human_review_required"):
        reasons.extend({"code": "semantic_uncertainty", "detail": str(detail)} for detail in verification.get("review_reasons") or ["核验员发现需要人工解释的适用条件。"])
    if verification.get("question_answered") is False and not reasons:
        reasons.append({"code": "incomplete_answer", "detail": "当前证据尚不足以完整回答所问事项。"})
    if verification.get("passed") is False:
        reasons.append({"code": "verification_failed", "detail": "自动核验未通过，需要检查候选结论与原始依据。"})
    reason = str(verification.get("reason") or "")
    if verification.get("finalization") or verification.get("stage") or reason.startswith(("correction_", "verification_unavailable")):
        reasons.append({"code": "processing_incomplete", "detail": "自动处理或校正未能完整完成；请结合阶段诊断审核。"})
    return list({(r["code"], r["detail"]): r for r in reasons}.values())


def prepare_review(state: dict[str, Any], candidate_answer: str) -> dict[str, Any] | None:
    reasons = review_reasons(state)
    if not reasons:
        return None
    messages = state.get("conversation_messages") or state.get("messages") or []
    question = state.get("rag_question") or next((m.content for m in reversed(messages) if m.role.value == "user"), "")
    return {
        "review_id": str(uuid.uuid4()), "status": "pending", "reasons": reasons,
        "task_id": (state.get("active_task") or {}).get("id"),
        "_snapshot": {
            "question": question, "task_request": state.get("rag_task_request") or question,
            "candidate_answer": candidate_answer, "draft": state.get("rag_draft"),
            "evidence": state.get("rag_evidence"), "citations": state.get("citations") or [],
            "verification": state.get("verification"), "stages": state.get("rag_stages") or [],
        },
    }


class ReviewConflict(Exception):
    pass


class HumanReviewStore:
    def __init__(self, redis_client: Any) -> None:
        self.redis = redis_client

    async def create(self, session_id: str, prepared: dict[str, Any]) -> dict[str, Any]:
        if self.redis is None:
            raise RuntimeError("Review storage unavailable")
        now = time.time()
        ticket = {k: v for k, v in prepared.items() if k != "_snapshot"}
        ticket.update(session_id=session_id, snapshot=prepared["_snapshot"], result_token=secrets.token_urlsafe(32),
                      created_at=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                      expires_at=datetime.fromtimestamp(now + settings.human_review_ttl_seconds, timezone.utc).isoformat(),
                      decision=None)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(review_key(ticket["review_id"]), json.dumps(ticket, ensure_ascii=False), ex=settings.human_review_ttl_seconds)
            pipe.zadd(INDEX, {ticket["review_id"]: now + settings.human_review_ttl_seconds})
            await pipe.execute()
        return {key: ticket.get(key) for key in ("review_id", "status", "reasons", "task_id", "created_at", "expires_at", "result_token")}

    async def get(self, review_id: str) -> dict[str, Any] | None:
        if self.redis is None:
            raise RuntimeError("Review storage unavailable")
        raw = await self.redis.get(review_key(review_id))
        return json.loads(raw) if raw else None

    async def pending(self, limit: int) -> list[dict[str, Any]]:
        if self.redis is None:
            raise RuntimeError("Review storage unavailable")
        # Expired tickets must not occupy the bounded queue page forever.
        await self.redis.zremrangebyscore(INDEX, "-inf", time.time())
        ids = await self.redis.zrange(INDEX, 0, limit - 1)
        tickets = []
        for review_id in ids:
            ticket = await self.get(review_id)
            if ticket and ticket["status"] == "pending":
                tickets.append(ticket)
            else:
                await self.redis.zrem(INDEX, review_id)
        return tickets

    async def decide(self, review_id: str, *, action: str, answer: str, notes: str, reviewer: str) -> dict[str, Any] | None:
        if self.redis is None:
            raise RuntimeError("Review storage unavailable")
        key = review_key(review_id)
        for _ in range(3):
            try:
                async with self.redis.pipeline(transaction=True) as pipe:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    if not raw:
                        return None
                    ticket = json.loads(raw)
                    if ticket["status"] != "pending":
                        raise ReviewConflict("该审核单已处理，不能覆盖已有审核决定。")
                    # The authenticated operator supplies the final answer explicitly.
                    # No language model rewrites an approved decision.
                    decision = {"reviewer": reviewer, "reviewed_at": datetime.now(timezone.utc).isoformat(),
                                "action": action, "notes": notes, "final_answer": answer if action == "approved" else ""}
                    ticket.update(status=action, decision=decision)
                    ttl = await pipe.ttl(key)
                    if ttl <= 0:
                        return None
                    session_key = "chat:session:" + ticket["session_id"]
                    await pipe.watch(session_key)
                    history_raw = await pipe.get(session_key)
                    history_ttl = await pipe.ttl(session_key)
                    history = json.loads(history_raw) if history_raw else None
                    if history is not None:
                        receipt = answer if action == "approved" else notes
                        history.append({"role": "assistant", "content": f"【人工审核：{action}】关于“{ticket['snapshot']['question']}”：\n{receipt}"})
                        history = history[-settings.memory_max_messages:]
                    pipe.multi()
                    pipe.set(key, json.dumps(ticket, ensure_ascii=False), ex=ttl)
                    pipe.zrem(INDEX, review_id)
                    if history is not None and history_ttl > 0:
                        pipe.set(session_key, json.dumps(history, ensure_ascii=False), ex=history_ttl)
                    await pipe.execute()
                    return ticket
            except WatchError:
                continue
        raise ReviewConflict("审核单正在被其他审核人处理，请刷新后重试。")


def public_result(ticket: dict[str, Any]) -> dict[str, Any]:
    decision = ticket.get("decision") or {}
    return {"review_id": ticket["review_id"], "status": ticket["status"], "reasons": ticket["reasons"],
            "created_at": ticket["created_at"], "expires_at": ticket["expires_at"],
            "final_answer": decision.get("final_answer", ""), "notes": decision.get("notes", ""),
            "reviewer": decision.get("reviewer"), "reviewed_at": decision.get("reviewed_at"),
            "question": ticket["snapshot"]["question"],
            "citations": ticket["snapshot"]["citations"] if ticket["status"] == "approved" else []}
