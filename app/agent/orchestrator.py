from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from app.config import settings
from app.core.tools.travel_search import (
    FlightSearchRequest,
    HotelSearchRequest,
    TrainSearchRequest,
    TravelRecommendationRequest,
    recommend_travel_options,
    search_flights,
    search_hotels,
    search_trains,
)
from app.domain.schemas import ChatMessage, MessageRole, StreamChunk, StreamChunkType
from app.domain.travel.itinerary import build_draft_itinerary, summarize_itinerary_text
from app.domain.travel.models import EmployeeGrade, TripPurpose, TravelRequest, TravelClass
from app.domain.travel.policy import apply_policy_to_itinerary, default_corporate_policy
from app.services.llm import LLMService

SYSTEM_PROMPT = """你是企业差旅助手「商旅-agent-guide」，帮助员工规划行程、解释差标与审批要求。
回答简洁专业，涉及金额与政策时标注「以公司制度为准」。
当用户要求查询、推荐或比较航班/高铁/火车/酒店时，优先调用搜索或推荐工具；正式预订前提醒用户确认库存、价格、退改规则和审批状态。
企业级输出要求：
1. 相对日期必须先解析成 YYYY-MM-DD 和星期，不得臆造月份。
2. 明确区分公司制度/RAG 命中、工具返回数据、模型推理建议、人工待确认项。
3. 若工具返回 provider=demo，必须标注“演示数据，非实时库存/价格”。
4. 推荐使用“结论摘要、行程方案、制度命中、差标校验、审批风险、数据来源与待确认事项”的结构。"""
SESSION_KEY_PREFIX = "chat:session"


def _travel_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "plan_travel_itinerary",
                "description": "根据结构化差旅需求生成草稿行程与费用预估。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "employee_id": {"type": "string"},
                        "grade": {
                            "type": "string",
                            "enum": [g.value for g in EmployeeGrade],
                        },
                        "origin_city": {"type": "string"},
                        "destination_city": {"type": "string"},
                        "departure_date": {"type": "string", "description": "YYYY-MM-DD"},
                        "return_date": {"type": "string", "description": "YYYY-MM-DD，可选"},
                        "purpose": {
                            "type": "string",
                            "enum": [p.value for p in TripPurpose],
                        },
                        "preferred_class": {
                            "type": "string",
                            "enum": [c.value for c in TravelClass],
                        },
                    },
                    "required": [
                        "employee_id",
                        "grade",
                        "origin_city",
                        "destination_city",
                        "departure_date",
                        "purpose",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_travel_policy",
                "description": "对已有行程草稿执行差标校验（舱位、预算、提前预订等）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "employee_id": {"type": "string"},
                        "grade": {
                            "type": "string",
                            "enum": [g.value for g in EmployeeGrade],
                        },
                        "origin_city": {"type": "string"},
                        "destination_city": {"type": "string"},
                        "departure_date": {"type": "string"},
                        "return_date": {"type": "string"},
                        "estimated_total_cny": {"type": "number"},
                        "preferred_class": {
                            "type": "string",
                            "enum": [c.value for c in TravelClass],
                        },
                    },
                    "required": [
                        "employee_id",
                        "grade",
                        "origin_city",
                        "destination_city",
                        "departure_date",
                        "estimated_total_cny",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_flights",
                "description": "查询指定城市和日期的航班候选，用于比价和推荐，正式预订前仍需确认库存。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "出发城市或 IATA 机场码"},
                        "destination": {"type": "string", "description": "到达城市或 IATA 机场码"},
                        "depart_date": {"type": "string", "description": "YYYY-MM-DD"},
                        "return_date": {"type": "string", "description": "YYYY-MM-DD，可选"},
                        "cabin": {
                            "type": "string",
                            "enum": [c.value for c in TravelClass],
                        },
                        "passengers": {"type": "integer", "minimum": 1, "maximum": 9},
                    },
                    "required": ["origin", "destination", "depart_date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_hotels",
                "description": "查询目的地酒店候选，支持按每晚预算和附近地点/商圈/客户地址过滤。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "check_in": {"type": "string", "description": "YYYY-MM-DD"},
                        "check_out": {"type": "string", "description": "YYYY-MM-DD"},
                        "keyword": {"type": "string"},
                        "poi_name": {
                            "type": "string",
                            "description": "酒店附近地点、商圈、车站、客户公司或 POI，如 陆家嘴、上海虹桥站、客户公司地址",
                        },
                        "guests": {"type": "integer", "minimum": 1, "maximum": 9},
                        "max_nightly_cny": {"type": "number"},
                    },
                    "required": ["city", "check_in", "check_out"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_trains",
                "description": (
                    "通过本地 12306 Skill 或 MCP 查询指定城市/车站和日期的真实火车/高铁余票候选。"
                    "如果 12306 查询不可用，必须明确说明未能查询，不得编造车次、票价或余票。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin_station": {"type": "string", "description": "出发城市或车站名，如 北京、北京南"},
                        "dest_station": {"type": "string", "description": "到达城市或车站名，如 上海、上海虹桥"},
                        "depart_date": {"type": "string", "description": "YYYY-MM-DD"},
                        "prefer_gd": {"type": "boolean", "description": "是否优先高铁/动车"},
                        "limited_num": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["origin_station", "dest_station", "depart_date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recommend_travel_options",
                "description": "综合查询航班、高铁/火车和酒店，并给出适合企业差旅的组合推荐。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "employee_id": {"type": "string"},
                        "grade": {
                            "type": "string",
                            "enum": [g.value for g in EmployeeGrade],
                        },
                        "origin_city": {"type": "string"},
                        "destination_city": {"type": "string"},
                        "departure_date": {"type": "string", "description": "YYYY-MM-DD"},
                        "return_date": {"type": "string", "description": "YYYY-MM-DD，可选"},
                        "purpose": {
                            "type": "string",
                            "enum": [p.value for p in TripPurpose],
                        },
                        "preferred_class": {
                            "type": "string",
                            "enum": [c.value for c in TravelClass],
                        },
                        "hotel_budget_cny": {"type": "number"},
                        "passenger_count": {"type": "integer", "minimum": 1, "maximum": 9},
                        "hotel_keyword": {"type": "string"},
                        "hotel_nearby_poi": {
                            "type": "string",
                            "description": "酒店位置约束：目的地附近地点、商圈、车站、客户公司或 POI，如 陆家嘴、上海虹桥站、客户公司地址",
                        },
                        "include_trains": {"type": "boolean", "description": "是否把高铁/火车候选纳入推荐，默认 true"},
                        "prefer_gd_trains": {"type": "boolean", "description": "高铁/火车候选是否优先 G/D 字头，默认 true"},
                    },
                    "required": [
                        "employee_id",
                        "grade",
                        "origin_city",
                        "destination_city",
                        "departure_date",
                    ],
                },
            },
        },
    ]


def _to_openai_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role.value, "content": m.content}
        if m.name:
            d["name"] = m.name
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        out.append(d)
    return out


def _parse_date(s: str) -> date:
    y, mo, d = (int(x) for x in s.split("-", 2))
    return date(y, mo, d)


def _today() -> date:
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


def _runtime_system_prompt(today: date | None = None) -> str:
    current = today or _today()
    weekday = "一二三四五六日"[current.weekday()]
    return (
        f"{SYSTEM_PROMPT}\n\n当前日期：{current.isoformat()}（周{weekday}，Asia/Shanghai）。"
        "所有“今天/明天/后天/本周/下周/周几”都必须以该日期为基准解析。"
    )


def _resolve_relative_date(text: str, *, today: date | None = None) -> date | None:
    base = today or _today()
    if "今天" in text:
        return base
    if "明天" in text:
        return base + timedelta(days=1)
    if "后天" in text:
        return base + timedelta(days=2)

    weekday_map = {
        "一": 0,
        "二": 1,
        "三": 2,
        "四": 3,
        "五": 4,
        "六": 5,
        "日": 6,
        "天": 6,
    }
    match = re.search(r"(下周|本周|这周)?周([一二三四五六日天])", text)
    if not match:
        return None
    prefix = match.group(1) or ""
    target = weekday_map[match.group(2)]
    if prefix == "下周":
        return base + timedelta(days=(7 - base.weekday()) + target)
    if prefix in {"本周", "这周"}:
        return base + timedelta(days=target - base.weekday())
    delta = target - base.weekday()
    if delta < 0:
        delta += 7
    return base + timedelta(days=delta)


def _weekday_label(day: date) -> str:
    return f"周{'一二三四五六日'[day.weekday()]}"


def _format_cn_date(day: date) -> str:
    return f"{day.year}年{day.month}月{day.day}日（{_weekday_label(day)}）"


def _extract_nearby_poi(text: str) -> str | None:
    patterns = (
        r"(?:酒店|住宿|住).{0,12}(?:安排在|在|靠近|离|距离)\s*([\u4e00-\u9fa5A-Za-z0-9·\-（）()]{2,30}?)(?:附近|周边)",
        r"(?:酒店|住宿|住).{0,12}(?:在|靠近|离|距离)?\s*([\u4e00-\u9fa5A-Za-z0-9·\-（）()]{2,30})(?:附近|周边)",
        r"(?:在|靠近|离|距离)\s*([\u4e00-\u9fa5A-Za-z0-9·\-（）()]{2,30})(?:附近|周边).{0,12}(?:酒店|住宿)",
        r"(?:客户|会议|办公|目的地|拜访地点).{0,8}(?:在|位于|靠近)?\s*([\u4e00-\u9fa5A-Za-z0-9·\-（）()]{2,30})(?:附近|周边)?",
    )
    stop_words = ("酒店", "住宿", "附近", "周边", "推荐", "查询", "安排")
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        value = match.group(1).strip(" ，。；;、")
        if value and not any(value == word for word in stop_words):
            return value
    return None


class TravelOrchestrator:
    def __init__(self, llm: Optional[LLMService] = None, redis_client: Any | None = None) -> None:
        self._llm = llm or LLMService()
        self._policy = default_corporate_policy()
        self._redis = redis_client

    def _session_key(self, session_id: str) -> str:
        return f"{SESSION_KEY_PREFIX}:{session_id}"

    @staticmethod
    def _message_identity(message: ChatMessage) -> tuple[str, str, str | None, str | None]:
        return (
            message.role.value,
            message.content,
            message.name,
            message.tool_call_id,
        )

    @classmethod
    def _merge_messages(
        cls,
        stored: list[ChatMessage],
        incoming: list[ChatMessage],
    ) -> list[ChatMessage]:
        if not stored:
            return incoming
        if not incoming:
            return stored

        max_overlap = min(len(stored), len(incoming))
        for overlap in range(max_overlap, 0, -1):
            stored_tail = [cls._message_identity(m) for m in stored[-overlap:]]
            incoming_head = [cls._message_identity(m) for m in incoming[:overlap]]
            if stored_tail == incoming_head:
                return stored + incoming[overlap:]

        if len(incoming) == 1:
            return stored + incoming
        return incoming

    def _trim_persisted_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        if len(messages) <= settings.memory_max_messages:
            return messages
        return messages[-settings.memory_max_messages :]

    async def _load_session_messages(self, session_id: str) -> list[ChatMessage]:
        if self._redis is None:
            return []
        raw = await self._redis.get(self._session_key(session_id))
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []

        out: list[ChatMessage] = []
        if not isinstance(payload, list):
            return out
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                out.append(ChatMessage.model_validate(item))
            except Exception:
                continue
        return out

    async def _save_session_messages(self, session_id: str, messages: list[ChatMessage]) -> None:
        if self._redis is None:
            return
        payload = json.dumps(
            [m.model_dump(mode="json") for m in self._trim_persisted_messages(messages)],
            ensure_ascii=False,
        )
        await self._redis.set(
            self._session_key(session_id),
            payload,
            ex=settings.memory_session_ttl_seconds,
        )

    async def _effective_messages(
        self,
        incoming_messages: list[ChatMessage],
        session_id: str | None,
    ) -> list[ChatMessage]:
        if not session_id:
            return incoming_messages
        stored_messages = await self._load_session_messages(session_id)
        return self._merge_messages(stored_messages, incoming_messages)

    async def _maybe_summarize_thread(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        if len(messages) <= settings.memory_summary_threshold:
            return messages
        head = messages[: -settings.memory_window_size]
        tail = messages[-settings.memory_window_size :]
        summary_req = [
            {
                "role": "system",
                "content": "将下列对话压缩为不超过200字的中文摘要，保留城市、日期、政策与金额。",
            },
            {"role": "user", "content": "\n".join(f"{m.role.value}: {m.content}" for m in head)},
        ]
        summary = await self._llm.chat_completion(summary_req, temperature=0.0)
        text = summary.choices[0].message.content or ""
        merged: List[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=f"[历史摘要] {text}")
        ]
        merged.extend(tail)
        return merged

    def _trim_window(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        if len(messages) <= settings.memory_window_size:
            return messages
        return messages[-settings.memory_window_size :]

    async def _execute_tool(self, name: str, arguments: str, user_text: str = "") -> str:
        args: dict[str, Any] = json.loads(arguments) if arguments else {}
        args = self._normalize_tool_args_from_text(args, user_text)

        if name == "plan_travel_itinerary":
            req = TravelRequest(
                request_id=str(uuid.uuid4()),
                employee_id=args["employee_id"],
                grade=EmployeeGrade(args["grade"]),
                origin_city=args["origin_city"],
                destination_city=args["destination_city"],
                departure_date=_parse_date(args["departure_date"]),
                return_date=_parse_date(args["return_date"]) if args.get("return_date") else None,
                purpose=TripPurpose(args["purpose"]),
                preferred_class=TravelClass(args["preferred_class"])
                if args.get("preferred_class")
                else None,
            )
            it = build_draft_itinerary(req)
            pc = req.preferred_class
            it = apply_policy_to_itinerary(self._policy, req, it, preferred_class=pc)
            return summarize_itinerary_text(it)

        if name == "check_travel_policy":
            dep = _parse_date(args["departure_date"])
            ret = _parse_date(args["return_date"]) if args.get("return_date") else None
            req = TravelRequest(
                request_id=str(uuid.uuid4()),
                employee_id=args["employee_id"],
                grade=EmployeeGrade(args["grade"]),
                origin_city=args["origin_city"],
                destination_city=args["destination_city"],
                departure_date=dep,
                return_date=ret,
                purpose=TripPurpose.CLIENT,
            )
            dummy = build_draft_itinerary(req)
            total = Decimal(str(args["estimated_total_cny"]))
            dummy = dummy.model_copy(update={"total_estimated_cny": total})
            pc = TravelClass(args["preferred_class"]) if args.get("preferred_class") else None
            checked = apply_policy_to_itinerary(self._policy, req, dummy, preferred_class=pc)
            return "差标校验结果：\n" + "\n".join(f"- {w}" for w in checked.policy_warnings)

        if name == "search_flights":
            result = await search_flights(FlightSearchRequest.model_validate(args))
            return json.dumps(result, ensure_ascii=False)

        if name == "search_hotels":
            result = await search_hotels(HotelSearchRequest.model_validate(args))
            return json.dumps(result, ensure_ascii=False)

        if name == "search_trains":
            result = await search_trains(TrainSearchRequest.model_validate(args))
            return json.dumps(result, ensure_ascii=False)

        if name == "recommend_travel_options":
            result = await recommend_travel_options(TravelRecommendationRequest.model_validate(args))
            return json.dumps(result, ensure_ascii=False)

        return json.dumps({"error": f"unknown tool {name}"}, ensure_ascii=False)

    @staticmethod
    def _normalize_tool_args_from_text(args: dict[str, Any], user_text: str) -> dict[str, Any]:
        if not user_text:
            return args
        normalized = dict(args)
        resolved = _resolve_relative_date(user_text)
        if resolved is not None:
            for key in ("departure_date", "depart_date"):
                if key in normalized:
                    normalized[key] = resolved.isoformat()
        if resolved is not None and "当天往返" in user_text and "return_date" in normalized:
            normalized["return_date"] = resolved.isoformat()
        poi = _extract_nearby_poi(user_text)
        if poi:
            if "destination_city" in normalized and not normalized.get("hotel_nearby_poi"):
                normalized["hotel_nearby_poi"] = poi
            if "city" in normalized and not normalized.get("poi_name"):
                normalized["poi_name"] = poi
        return normalized

    @staticmethod
    def _last_incoming_user_text(messages: list[ChatMessage]) -> str:
        for msg in reversed(messages):
            if msg.role is MessageRole.USER:
                return msg.content
        return ""

    @staticmethod
    def _uses_demo_inventory(tool_trace: list[dict[str, Any]]) -> bool:
        for item in tool_trace:
            output = item.get("output") if isinstance(item, dict) else None
            if not isinstance(output, str):
                continue
            try:
                payload = json.loads(output)
            except json.JSONDecodeError:
                payload = {}
            if payload.get("provider") == "demo" or "demo provider" in output:
                return True
            for key in ("results", "flights", "hotels"):
                values = payload.get(key)
                if isinstance(values, list) and any(
                    isinstance(value, dict) and value.get("provider") == "demo" for value in values
                ):
                    return True
        return False

    @staticmethod
    def _policy_warnings_from_tool_trace(tool_trace: list[dict[str, Any]]) -> list[str]:
        warnings: list[str] = []
        for item in tool_trace:
            output = item.get("output") if isinstance(item, dict) else None
            if not isinstance(output, str):
                continue
            try:
                payload = json.loads(output)
            except json.JSONDecodeError:
                payload = {}
            values = payload.get("policy_warnings") if isinstance(payload, dict) else None
            if isinstance(values, list):
                for value in values:
                    warning = str(value).strip()
                    if warning and warning not in warnings:
                        warnings.append(warning)
        return warnings

    @staticmethod
    def _warning_already_covered(answer: str, warning: str) -> bool:
        if warning in answer:
            return True
        compact_answer = re.sub(r"\s+", "", answer)
        compact_warning = re.sub(r"\s+", "", warning)
        if "低于提前7天" in compact_warning:
            return "低于提前7天" in compact_answer or "提前7天" in compact_answer and "附加费" in compact_answer
        return False

    @staticmethod
    def _enforce_enterprise_answer_contract(
        answer: str,
        tool_trace: list[dict[str, Any]],
        messages: list[ChatMessage],
    ) -> str:
        user_text = TravelOrchestrator._last_incoming_user_text(messages)
        resolved = _resolve_relative_date(user_text)
        updated = answer
        if resolved is not None and any(term in user_text for term in ("下周", "本周", "这周", "周", "明天", "后天", "今天")):
            expected = _format_cn_date(resolved)
            updated = re.sub(r"\d{1,2}月\d{1,2}日（下周[一二三四五六日天]?）", expected, updated)
            updated = re.sub(r"\d{4}年\d{1,2}月\d{1,2}日（下周[一二三四五六日天]?）", expected, updated)
            if resolved.isoformat() not in updated and expected not in updated:
                updated = f"日期校验：按当前日期 {_today().isoformat()} 解析，用户所述相对日期对应 {expected}。\n\n{updated}"

        if TravelOrchestrator._uses_demo_inventory(tool_trace) and not any(
            term in updated for term in ("演示数据", "demo", "非实时库存", "非真实库存")
        ):
            updated = (
                f"{updated}\n\n数据来源说明：航班/酒店候选来自本地 demo provider，"
                "属于演示数据，非实时库存或最终价格；正式预订前必须接入真实供应商或人工确认。"
            )
        policy_warnings = TravelOrchestrator._policy_warnings_from_tool_trace(tool_trace)
        missing_warnings = [
            warning
            for warning in policy_warnings
            if not TravelOrchestrator._warning_already_covered(updated, warning)
        ]
        if missing_warnings:
            updated = (
                f"{updated}\n\n差标/审批校验补充："
                + "；".join(missing_warnings)
            )
        return updated

    async def run_completion(
        self,
        messages: list[ChatMessage],
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        _ = user_id
        effective_messages = await self._effective_messages(messages, session_id)
        msgs = await self._maybe_summarize_thread(effective_messages)
        msgs = self._trim_window(msgs)
        openai_msgs: List[Dict[str, Any]] = [{"role": "system", "content": _runtime_system_prompt()}]
        openai_msgs.extend(_to_openai_messages(msgs))

        tools = _travel_tools()
        tool_trace: list[dict[str, Any]] = []
        for _ in range(settings.max_react_iterations):
            resp = await self._llm.chat_completion(openai_msgs, tools=tools, tool_choice="auto")
            choice = resp.choices[0]
            msg = choice.message

            if msg.tool_calls:
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                openai_msgs.append(assistant_msg)
                for tc in msg.tool_calls:
                    out = await self._execute_tool(
                        tc.function.name,
                        tc.function.arguments,
                        self._last_incoming_user_text(effective_messages),
                    )
                    tool_trace.append(
                        {
                            "tool": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                            "output": out,
                        }
                    )
                    openai_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": out,
                        }
                    )
                continue

            content = msg.content or ""
            content = self._enforce_enterprise_answer_contract(content, tool_trace, effective_messages)
            if session_id:
                await self._save_session_messages(
                    session_id,
                    effective_messages
                    + [ChatMessage(role=MessageRole.ASSISTANT, content=content)],
                )
            return {
                "id": getattr(resp, "id", str(uuid.uuid4())),
                "created": int(time.time()),
                "model": self._llm.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
                "usage": {
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                    "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                },
            }

        return {
            "id": str(uuid.uuid4()),
            "created": int(time.time()),
            "model": self._llm.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "已达到最大推理轮次，请简化问题后重试。",
                    },
                }
            ],
            "usage": None,
        }

    async def stream_completion(
        self,
        messages: list[ChatMessage],
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        result = await self.run_completion(messages, session_id=session_id, user_id=user_id)
        text = result["choices"][0]["message"]["content"]
        for i, ch in enumerate(text):
            yield StreamChunk(type=StreamChunkType.CONTENT, index=i, delta=ch)
        yield StreamChunk(
            type=StreamChunkType.DONE,
            index=len(text),
            finish_reason="stop",
        )
