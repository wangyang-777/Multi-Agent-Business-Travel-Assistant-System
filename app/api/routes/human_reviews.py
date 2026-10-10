"""Internal operator queue plus a session-scoped public decision receipt."""
from __future__ import annotations

import secrets
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, model_validator
from redis.exceptions import RedisError

from app.config import settings
from app.services.human_review import HumanReviewStore, ReviewConflict, public_result

router = APIRouter(prefix="/human-reviews", tags=["human reviews"])


class ReviewDecision(BaseModel):
    action: Literal["approved", "rejected", "needs_information"]
    final_answer: str = Field(default="", max_length=20000)
    notes: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def validate_decision(self) -> "ReviewDecision":
        self.notes = self.notes.strip()
        self.final_answer = self.final_answer.strip()
        if not self.notes or (self.action == "approved" and not self.final_answer):
            raise ValueError("需填写审核理由；通过时必须填写明确的最终答复。")
        if self.action != "approved" and self.final_answer:
            raise ValueError("未通过审核时不得发布最终答复。")
        return self


def reviewer(authorization: str | None = Header(default=None)) -> str:
    token = settings.human_review_api_token
    if not token:
        raise HTTPException(503, "审核入口尚未配置，请配置 HUMAN_REVIEW_API_TOKEN 后重启服务。")
    supplied = (authorization or "").removeprefix("Bearer ")
    if not (authorization or "").startswith("Bearer ") or not secrets.compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(401, "审核凭据无效。", headers={"WWW-Authenticate": "Bearer"})
    return settings.human_review_reviewer_name


def store(request: Request, response: Response) -> HumanReviewStore:
    response.headers["Cache-Control"] = "no-store"
    client = getattr(request.app.state, "redis", None)
    if client is None:
        raise HTTPException(503, "审核存储不可用，请稍后重试。")
    return HumanReviewStore(client)


@router.get("")
async def pending_reviews(limit: int = Query(default=50, ge=1, le=100), actor: str = Depends(reviewer), repository: HumanReviewStore = Depends(store)) -> dict[str, Any]:
    try:
        tickets = await repository.pending(limit)
    except RedisError:
        raise HTTPException(503, "审核存储暂时不可用。") from None
    return {"reviewer": actor, "reviews": [{"review_id": t["review_id"], "question": t["snapshot"]["question"], "task_request": t["snapshot"]["task_request"], "reasons": t["reasons"], "created_at": t["created_at"]} for t in tickets]}


@router.get("/{review_id}/result")
async def review_result(review_id: UUID, session_id: str = Query(min_length=1), result_token: str | None = Header(default=None, alias="X-Review-Result-Token"), repository: HumanReviewStore = Depends(store)) -> dict[str, Any]:
    try:
        ticket = await repository.get(str(review_id))
    except RedisError:
        raise HTTPException(503, "审核存储暂时不可用。") from None
    if not ticket or ticket["session_id"] != session_id or not result_token or not secrets.compare_digest(result_token.encode(), ticket["result_token"].encode()):
        raise HTTPException(404, "审核单不存在或不属于当前会话。")
    return public_result(ticket)


@router.get("/{review_id}")
async def review_detail(review_id: UUID, actor: str = Depends(reviewer), repository: HumanReviewStore = Depends(store)) -> dict[str, Any]:
    try:
        ticket = await repository.get(str(review_id))
    except RedisError:
        raise HTTPException(503, "审核存储暂时不可用。") from None
    if not ticket:
        raise HTTPException(404, "审核单不存在或已过期。")
    return {key: value for key, value in ticket.items() if key != "result_token"}


@router.post("/{review_id}/decision")
async def decide_review(review_id: UUID, body: ReviewDecision, actor: str = Depends(reviewer), repository: HumanReviewStore = Depends(store)) -> dict[str, Any]:
    try:
        ticket = await repository.decide(str(review_id), action=body.action, answer=body.final_answer, notes=body.notes, reviewer=actor)
    except ReviewConflict as exc:
        raise HTTPException(409, str(exc)) from None
    except RedisError:
        raise HTTPException(503, "审核存储暂时不可用，请查询审核单状态后再重试。") from None
    if not ticket:
        raise HTTPException(404, "审核单不存在或已过期。")
    return public_result(ticket)
