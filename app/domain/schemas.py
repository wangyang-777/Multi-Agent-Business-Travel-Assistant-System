from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, description="对话消息列表")
    stream: bool = False
    session_id: Optional[str] = Field(None, description="会话 ID，用于记忆与审计")
    user_id: Optional[str] = Field(None, description="企业用户标识")
    locale: str = Field("zh-CN", description="语言区域")


class ToolCallInfo(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ResponseTable(BaseModel):
    title: str
    columns: list[str]
    rows: list[dict[str, str]]


class ResponseCitation(BaseModel):
    chunk_id: Optional[str] = None
    title: Optional[str] = None
    doc_type: Optional[str] = None
    content: str
    score: Optional[float] = None
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalForm(BaseModel):
    required: bool = False
    status: Literal["not_required", "pending_human_approval"] = "not_required"
    employee_id: Optional[str] = None
    grade: Optional[str] = None
    origin_city: Optional[str] = None
    destination_city: Optional[str] = None
    departure_date: Optional[str] = None
    return_date: Optional[str] = None
    estimated_total_cny: Optional[str] = None
    reason: Optional[str] = None
    policy_warnings: list[str] = Field(default_factory=list)


class BookingDraft(BaseModel):
    draft_id: str
    status: Literal["draft_created", "approval_pending"] = "draft_created"
    employee_id: Optional[str] = None
    grade: Optional[str] = None
    origin_city: Optional[str] = None
    destination_city: Optional[str] = None
    departure_date: Optional[str] = None
    return_date: Optional[str] = None
    purpose: Optional[str] = None
    recommended_flight: dict[str, Any] = Field(default_factory=dict)
    recommended_train: dict[str, Any] = Field(default_factory=dict)
    recommended_hotel: dict[str, Any] = Field(default_factory=dict)
    price_snapshot: dict[str, str] = Field(default_factory=dict)
    estimated_total_cny: Optional[str] = None
    policy_checks: dict[str, Any] = Field(default_factory=dict)
    policy_warnings: list[str] = Field(default_factory=list)
    approval_required: bool = False
    confirmation_items: list[str] = Field(default_factory=list)
    next_action: str = "confirm_draft"
    created_at: str
    expires_at: str


class ChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: Optional[dict[str, int]] = None
    session_id: Optional[str] = None
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    tables: list[ResponseTable] = Field(default_factory=list)
    citations: list[ResponseCitation] = Field(default_factory=list)
    approval_form: Optional[ApprovalForm] = None
    booking_draft: Optional[BookingDraft] = None
    execution_plan: Optional[dict[str, Any]] = None
    policy_constraints: Optional[dict[str, Any]] = None
    policy_validation: Optional[dict[str, Any]] = None
    travel_retry: Optional[dict[str, Any]] = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    risk_level: Optional[Literal["low", "medium", "high"]] = None
    answer_mode: Optional[Literal["rag_grounded", "llm_fallback"]] = None
    verification: Optional[dict[str, Any]] = None
    claim_evidence_map: list[dict[str, Any]] = Field(default_factory=list)
    rag_correction_count: int = 0


class StreamChunkType(str, Enum):
    CONTENT = "content"
    TOOL_CALL = "tool_call"
    DONE = "done"
    ERROR = "error"


class StreamChunk(BaseModel):
    type: StreamChunkType
    index: int = 0
    delta: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None


class DocumentIngestRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., min_length=1, description="纯文本正文，用于向量检索")
    doc_type: Literal["policy", "sop", "city_guide", "other"] = "policy"
    metadata: dict[str, Any] = Field(default_factory=dict)


class LongDocumentIngestRequest(DocumentIngestRequest):
    chunk_size: int = Field(
        3000,
        ge=200,
        le=12000,
        description="语义单元异常过大时使用的字符安全上限，不作为常规固定切分长度",
    )
    chunk_overlap: int = Field(
        450,
        ge=0,
        le=2000,
        description="兼容旧客户端保留；在线分块固定按前一语义 chunk 的 15% 计算 overlap",
    )


class DocumentBatchDeleteRequest(BaseModel):
    doc_ids: list[str] = Field(..., min_length=1, max_length=500)


class DocumentIngestResponse(BaseModel):
    doc_id: str
    collection: str
    inserted_at: datetime
    vector_dim: Optional[int] = None
    chunk_count: int = 1
    chunk_ids: list[str] = Field(default_factory=list)


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded", "unhealthy"]
    version: str = "0.1.0"
    checks: dict[str, bool] = Field(default_factory=dict)
    detail: Optional[str] = None
