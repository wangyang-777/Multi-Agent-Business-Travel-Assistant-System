from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any, Union

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.agent.orchestrator import TravelOrchestrator
from app.domain.schemas import (
    ApprovalForm,
    BookingDraft,
    ChatRequest,
    ChatResponse,
    ResponseCitation,
    ResponseTable,
    StreamChunk,
    StreamChunkType,
)

router = APIRouter(tags=["chat"])


def get_orchestrator(request: Request) -> TravelOrchestrator:
    return request.app.state.orchestrator


_SEGMENT_RE = re.compile(
    r"^(?P<index>\d+)\.\s+(?P<kind>\S+)\s+(?P<route>\S+)\s+"
    r"(?P<depart>\S+)\s+[—-]\s+(?P<arrive>\S+)$"
)


def _table_from_itinerary_text(text: str) -> list[ResponseTable]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    tables: list[ResponseTable] = []
    overview_rows: list[dict[str, str]] = []
    segment_rows: list[dict[str, str]] = []
    warning_rows: list[dict[str, str]] = []

    for line in lines:
        if line.startswith("【") and line.endswith("】"):
            overview_rows.append({"项目": "行程标题", "内容": line.strip("【】")})
            continue
        if line.startswith("预估总额："):
            overview_rows.append({"项目": "预估总额", "内容": line.removeprefix("预估总额：")})
            continue
        if line.startswith("策略提示："):
            warnings = [w.strip() for w in line.removeprefix("策略提示：").split("；") if w.strip()]
            warning_rows.extend(
                {"序号": str(i), "提示": warning} for i, warning in enumerate(warnings, start=1)
            )
            continue

        match = _SEGMENT_RE.match(line)
        if match:
            segment_rows.append(
                {
                    "序号": match.group("index"),
                    "类型": match.group("kind"),
                    "路线": match.group("route"),
                    "出发时间": match.group("depart"),
                    "到达/结束时间": match.group("arrive"),
                }
            )

    if overview_rows:
        tables.append(
            ResponseTable(
                title="行程概览",
                columns=["项目", "内容"],
                rows=overview_rows,
            )
        )
    if segment_rows:
        tables.append(
            ResponseTable(
                title="行程明细",
                columns=["序号", "类型", "路线", "出发时间", "到达/结束时间"],
                rows=segment_rows,
            )
        )
    if warning_rows:
        tables.append(
            ResponseTable(
                title="差标与策略提示",
                columns=["序号", "提示"],
                rows=warning_rows,
            )
        )
    return tables


def _split_markdown_table_row(line: str) -> list[str]:
    text = line.strip()
    if not (text.startswith("|") and text.endswith("|")):
        return []
    return [cell.strip().replace("<br>", "\n") for cell in text.strip("|").split("|")]


def _is_markdown_separator(line: str) -> bool:
    cells = _split_markdown_table_row(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _table_from_markdown_text(text: str) -> list[ResponseTable]:
    lines = text.splitlines()
    tables: list[ResponseTable] = []
    index = 0
    table_no = 1

    while index < len(lines) - 1:
        header = _split_markdown_table_row(lines[index])
        if not header or not _is_markdown_separator(lines[index + 1]):
            index += 1
            continue

        title = f"正文表格 {table_no}"
        for prev in range(index - 1, max(-1, index - 6), -1):
            heading = lines[prev].strip().strip("#").strip()
            if heading:
                title = heading[:40]
                break

        rows: list[dict[str, str]] = []
        row_index = index + 2
        while row_index < len(lines):
            cells = _split_markdown_table_row(lines[row_index])
            if not cells:
                break
            normalized = cells[: len(header)] + [""] * max(0, len(header) - len(cells))
            rows.append({col: normalized[pos] for pos, col in enumerate(header)})
            row_index += 1

        if rows:
            tables.append(ResponseTable(title=title, columns=header, rows=rows))
            table_no += 1
        index = max(row_index, index + 2)

    return tables


def _table_from_booking_draft(draft: dict[str, Any]) -> list[ResponseTable]:
    if not draft:
        return []
    flight = draft.get("recommended_flight") if isinstance(draft.get("recommended_flight"), dict) else {}
    train = draft.get("recommended_train") if isinstance(draft.get("recommended_train"), dict) else {}
    hotel = draft.get("recommended_hotel") if isinstance(draft.get("recommended_hotel"), dict) else {}
    route = f"{_string(draft.get('origin_city'))} → {_string(draft.get('destination_city'))}"
    rows = [
        {"项目": "草稿 ID", "内容": _string(draft.get("draft_id"))},
        {"项目": "状态", "内容": _string(draft.get("status"))},
        {"项目": "路线", "内容": route},
        {"项目": "日期", "内容": _string(draft.get("departure_date"))},
        {"项目": "推荐航班", "内容": _string(flight.get("flight_no"))},
        {"项目": "推荐高铁/火车", "内容": _string(train.get("train_code"))},
        {"项目": "推荐酒店", "内容": _string(hotel.get("name"))},
        {"项目": "预估金额", "内容": _string(draft.get("estimated_total_cny") or "待确认")},
        {"项目": "是否需要审批", "内容": "是" if draft.get("approval_required") else "否"},
        {"项目": "下一步", "内容": _string(draft.get("next_action"))},
    ]
    tables = [
        ResponseTable(
            title="预订草稿",
            columns=["项目", "内容"],
            rows=rows,
        )
    ]
    items = [str(item) for item in draft.get("confirmation_items") or [] if str(item).strip()]
    if items:
        tables.append(
            ResponseTable(
                title="预订确认项",
                columns=["序号", "确认项"],
                rows=[
                    {"序号": str(index), "确认项": item}
                    for index, item in enumerate(items, start=1)
                ],
            )
        )
    return tables


def _table_from_execution_plan(plan: dict[str, Any]) -> list[ResponseTable]:
    if not plan:
        return []
    summary_rows = [
        {"项目": "目标", "内容": _string(plan.get("goal"))},
        {"项目": "Planner", "内容": _string(plan.get("planner"))},
        {"项目": "需要工具", "内容": "、".join(str(item) for item in plan.get("required_tools") or [])},
        {"项目": "缺失槽位", "内容": "、".join(str(item) for item in plan.get("missing_slots") or []) or "无"},
        {"项目": "是否需澄清", "内容": "是" if plan.get("needs_clarification") else "否"},
        {"项目": "规划依据", "内容": _string(plan.get("rationale"))},
    ]
    tables = [
        ResponseTable(
            title="执行计划",
            columns=["项目", "内容"],
            rows=summary_rows,
        )
    ]
    steps = [item for item in plan.get("steps") or [] if isinstance(item, dict)]
    if steps:
        tables.append(
            ResponseTable(
                title="执行步骤",
                columns=["顺序", "工具", "说明"],
                rows=[
                    {
                        "顺序": _string(item.get("order") or index),
                        "工具": _string(item.get("tool") or "-"),
                        "说明": _string(item.get("reason") or item.get("task")),
                    }
                    for index, item in enumerate(steps, start=1)
                ],
            )
        )
    return tables


def _table_from_policy_constraints(payload: dict[str, Any]) -> list[ResponseTable]:
    if not payload:
        return []
    constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
    notes = payload.get("notes") if isinstance(payload.get("notes"), list) else []
    labels = {
        "hotel_limit_cny": "酒店标准上限",
        "advance_booking_days": "提前预订天数",
        "approval_threshold_cny": "审批金额阈值",
        "cabin_limit": "航班舱位约束",
        "train_seat": "高铁/火车席别",
    }
    rows = [
        {"项目": "来源", "内容": _string(payload.get("source"))},
        {"项目": "置信度", "内容": _string(payload.get("confidence"))},
    ]
    rows.extend(
        {"项目": labels.get(key, key), "内容": _string(value)}
        for key, value in constraints.items()
    )
    if notes:
        rows.append({"项目": "备注", "内容": "；".join(str(item) for item in notes)})
    return [
        ResponseTable(
            title="制度约束",
            columns=["项目", "内容"],
            rows=rows,
        )
    ]


def _table_from_policy_validation(validation: dict[str, Any]) -> list[ResponseTable]:
    if not validation:
        return []
    tables = [
        ResponseTable(
            title="合规校验",
            columns=["项目", "内容"],
            rows=[
                {"项目": "状态", "内容": _string(validation.get("status"))},
                {"项目": "摘要", "内容": _string(validation.get("summary"))},
                {"项目": "制度来源", "内容": _string(validation.get("constraint_source"))},
                {"项目": "约束置信度", "内容": _string(validation.get("constraint_confidence"))},
            ],
        )
    ]
    checks = [item for item in validation.get("checks") or [] if isinstance(item, dict)]
    if checks:
        tables.append(
            ResponseTable(
                title="合规检查项",
                columns=["检查项", "状态", "说明"],
                rows=[
                    {
                        "检查项": _string(item.get("name")),
                        "状态": _string(item.get("status")),
                        "说明": _string(item.get("detail")),
                    }
                    for item in checks
                ],
            )
        )
    issues = [
        {"类型": "风险", "内容": str(item)}
        for item in validation.get("violations") or []
        if str(item).strip()
    ]
    issues.extend(
        {"类型": "提示", "内容": str(item)}
        for item in validation.get("warnings") or []
        if str(item).strip()
    )
    if issues:
        tables.append(
            ResponseTable(
                title="合规风险与提示",
                columns=["类型", "内容"],
                rows=issues,
            )
        )
    return tables


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _table_from_inventory_json(text: str) -> list[ResponseTable]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []

    mode = payload.get("mode")
    tables: list[ResponseTable] = []

    def flight_rows(results: list[dict[str, Any]]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for item in results:
            rows.append(
                {
                    "航班": _string(item.get("flight_no")),
                    "航司": _string(item.get("carrier")),
                    "路线": f"{_string(item.get('origin'))} → {_string(item.get('destination'))}",
                    "出发": _string(item.get("depart_at")),
                    "到达": _string(item.get("arrive_at")),
                    "舱位": _string(item.get("cabin")),
                    "价格": _string(item.get("price_cny")),
                    "预订链接": _string(item.get("booking_url")),
                    "推荐理由": _string(item.get("reason")),
                }
            )
        return rows

    def hotel_rows(results: list[dict[str, Any]]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for item in results:
            rows.append(
                {
                    "酒店": _string(item.get("name")),
                    "城市": _string(item.get("city")),
                    "位置约束": _string(item.get("poi_name")),
                    "周边": _string(item.get("nearby")),
                    "星级": _string(item.get("star")),
                    "入住": _string(item.get("check_in")),
                    "离店": _string(item.get("check_out")),
                    "每晚": _string(item.get("nightly_cny")),
                    "总价": _string(item.get("total_cny")),
                    "详情链接": _string(item.get("detail_url")),
                    "推荐理由": _string(item.get("reason")),
                }
            )
        return rows

    def train_rows(results: list[dict[str, Any]]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for item in results:
            rows.append(
                {
                    "车次": _string(item.get("trainCode") or item.get("train_code")),
                    "路线": (
                        f"{_string(item.get('fromStation') or item.get('from_station'))}"
                        f" → {_string(item.get('toStation') or item.get('to_station'))}"
                    ),
                    "出发": _string(item.get("departTime") or item.get("depart_time")),
                    "到达": _string(item.get("arriveTime") or item.get("arrive_time")),
                    "耗时": _string(item.get("duration")),
                    "商务/特等": _string(item.get("swz") or item.get("tz")),
                    "一等座": _string(item.get("zy")),
                    "二等座": _string(item.get("ze")),
                    "状态": "可购" if item.get("canBuy") == "Y" else _string(item.get("canBuy")),
                    "推荐分": _string(item.get("recommendation_score")),
                    "推荐理由": _string(item.get("reason")),
                }
            )
        return rows

    if mode == "flight":
        rows = flight_rows([r for r in payload.get("results", []) if isinstance(r, dict)])
        if rows:
            tables.append(
                ResponseTable(
                    title="航班候选",
                    columns=["航班", "航司", "路线", "出发", "到达", "舱位", "价格", "预订链接", "推荐理由"],
                    rows=rows,
                )
            )
    elif mode == "hotel":
        rows = hotel_rows([r for r in payload.get("results", []) if isinstance(r, dict)])
        if rows:
            tables.append(
                ResponseTable(
                    title="酒店候选",
                    columns=["酒店", "城市", "位置约束", "周边", "星级", "入住", "离店", "每晚", "总价", "详情链接", "推荐理由"],
                    rows=rows,
                )
            )
    elif mode == "train":
        results = [r for r in payload.get("results", []) if isinstance(r, dict)]
        query = payload.get("query") if isinstance(payload.get("query"), dict) else {}
        summary_rows = [
            {"项目": "路线", "内容": f"{_string(query.get('origin_station'))} → {_string(query.get('dest_station'))}"},
            {"项目": "日期", "内容": _string(query.get("depart_date"))},
            {"项目": "数据源", "内容": _string(payload.get("provider"))},
            {"项目": "说明", "内容": _string(payload.get("disclaimer"))},
        ]
        if payload.get("error"):
            summary_rows.append({"项目": "错误", "内容": _string(payload.get("error"))})
        tables.append(
            ResponseTable(
                title="12306 查询摘要",
                columns=["项目", "内容"],
                rows=summary_rows,
            )
        )
        rows = train_rows(results)
        if rows:
            tables.append(
                ResponseTable(
                    title="12306 车票候选",
                    columns=["车次", "路线", "出发", "到达", "耗时", "商务/特等", "一等座", "二等座", "状态", "推荐分", "推荐理由"],
                    rows=rows,
                )
            )
    elif mode == "travel_recommendation":
        flights = [r for r in payload.get("flights", []) if isinstance(r, dict)]
        trains = [r for r in payload.get("trains", []) if isinstance(r, dict)]
        hotels = [r for r in payload.get("hotels", []) if isinstance(r, dict)]
        rec = payload.get("recommendation") if isinstance(payload.get("recommendation"), dict) else {}
        rec_flight = rec.get("flight") if isinstance(rec.get("flight"), dict) else {}
        rec_train = rec.get("train") if isinstance(rec.get("train"), dict) else {}
        rec_hotel = rec.get("hotel") if isinstance(rec.get("hotel"), dict) else {}
        policy_warnings = [str(w) for w in payload.get("policy_warnings", []) if w]
        policy_checks = payload.get("policy_checks") if isinstance(payload.get("policy_checks"), dict) else {}
        tables.append(
            ResponseTable(
                title="推荐组合",
                columns=["项目", "内容"],
                rows=[
                    {"项目": "推荐航班", "内容": _string(rec_flight.get("flight_no"))},
                    {"项目": "推荐高铁/火车", "内容": _string(rec_train.get("trainCode") or rec_train.get("train_code"))},
                    {"项目": "推荐酒店", "内容": _string(rec_hotel.get("name"))},
                    {"项目": "推荐原因", "内容": _string(rec.get("reason"))},
                    {
                        "项目": "提前预订",
                        "内容": (
                            "满足要求"
                            if policy_checks.get("advance_booking_ok")
                            else "低于提前预订要求"
                        ),
                    },
                    {"项目": "说明", "内容": _string(payload.get("disclaimer"))},
                ],
            )
        )
        if policy_warnings:
            tables.append(
                ResponseTable(
                    title="差标与审批风险",
                    columns=["序号", "提示"],
                    rows=[
                        {"序号": str(index), "提示": warning}
                        for index, warning in enumerate(policy_warnings, start=1)
                    ],
                )
            )
        flight_table_rows = flight_rows(flights)
        train_table_rows = train_rows(trains)
        hotel_table_rows = hotel_rows(hotels)
        if flight_table_rows:
            tables.append(
                ResponseTable(
                    title="航班候选",
                    columns=["航班", "航司", "路线", "出发", "到达", "舱位", "价格", "预订链接", "推荐理由"],
                    rows=flight_table_rows,
                )
            )
        if train_table_rows:
            tables.append(
                ResponseTable(
                    title="高铁/火车候选",
                    columns=["车次", "路线", "出发", "到达", "耗时", "商务/特等", "一等座", "二等座", "状态", "推荐分", "推荐理由"],
                    rows=train_table_rows,
                )
            )
        if hotel_table_rows:
            tables.append(
                ResponseTable(
                    title="酒店候选",
                    columns=["酒店", "城市", "位置约束", "周边", "星级", "入住", "离店", "每晚", "总价", "详情链接", "推荐理由"],
                    rows=hotel_table_rows,
                )
            )

    return tables


def _build_response_tables(raw: dict[str, Any]) -> list[ResponseTable]:
    tables: list[ResponseTable] = []
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    tool_trace = metadata.get("tool_trace") if isinstance(metadata, dict) else None
    execution_plan = metadata.get("execution_plan") if isinstance(metadata, dict) else None
    policy_constraints = metadata.get("policy_constraints") if isinstance(metadata, dict) else None
    policy_validation = metadata.get("policy_validation") if isinstance(metadata, dict) else None
    booking_draft = metadata.get("booking_draft") if isinstance(metadata, dict) else None

    if isinstance(execution_plan, dict):
        tables.extend(_table_from_execution_plan(execution_plan))
    if isinstance(policy_constraints, dict):
        tables.extend(_table_from_policy_constraints(policy_constraints))
    if isinstance(policy_validation, dict):
        tables.extend(_table_from_policy_validation(policy_validation))
    if isinstance(booking_draft, dict):
        tables.extend(_table_from_booking_draft(booking_draft))

    if isinstance(tool_trace, list):
        attempt_values = [
            int(item["attempt"])
            for item in tool_trace
            if isinstance(item, dict) and isinstance(item.get("attempt"), int)
        ]
        display_attempt = max(attempt_values) if attempt_values else None
        for item in tool_trace:
            if not isinstance(item, dict):
                continue
            if display_attempt is not None and item.get("attempt") != display_attempt:
                continue
            output = item.get("output")
            if isinstance(output, str):
                tables.extend(_table_from_inventory_json(output))
                tables.extend(_table_from_itinerary_text(output))

    if tables:
        return tables

    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return []
    if not isinstance(content, str):
        return []
    return _table_from_itinerary_text(content) + _table_from_markdown_text(content)


def _metadata(raw: dict[str, Any]) -> dict[str, Any]:
    metadata = raw.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _build_citations(raw: dict[str, Any]) -> list[ResponseCitation]:
    citations = _metadata(raw).get("citations")
    if not isinstance(citations, list):
        return []
    out: list[ResponseCitation] = []
    for item in citations:
        if not isinstance(item, dict):
            continue
        out.append(ResponseCitation.model_validate(item))
    return out


def _build_approval_form(raw: dict[str, Any]) -> ApprovalForm | None:
    form = _metadata(raw).get("approval_form")
    if not isinstance(form, dict):
        return None
    return ApprovalForm.model_validate(form)


def _build_booking_draft(raw: dict[str, Any]) -> BookingDraft | None:
    draft = _metadata(raw).get("booking_draft")
    if not isinstance(draft, dict):
        return None
    return BookingDraft.model_validate(draft)


@router.post("/chat", response_model=None)
async def chat(
    body: ChatRequest,
    request: Request,
    orchestrator: TravelOrchestrator = Depends(get_orchestrator),
) -> Union[ChatResponse, StreamingResponse]:
    if body.stream:

        async def event_gen() -> AsyncIterator[str]:
            idx = 0
            try:
                async for chunk in orchestrator.stream_completion(
                    body.messages,
                    session_id=body.session_id,
                    user_id=body.user_id,
                ):
                    payload = chunk.model_dump(mode="json")
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    idx += 1
            except Exception as exc:
                err = StreamChunk(
                    type=StreamChunkType.ERROR,
                    index=idx,
                    error=str(exc),
                )
                yield f"data: {json.dumps(err.model_dump(mode='json'), ensure_ascii=False)}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    raw: dict[str, Any] = await orchestrator.run_completion(
        body.messages,
        session_id=body.session_id,
        user_id=body.user_id,
    )
    metadata = _metadata(raw)
    return ChatResponse(
        id=raw["id"],
        created=raw["created"],
        model=raw["model"],
        choices=raw["choices"],
        usage=raw.get("usage"),
        session_id=body.session_id,
        tables=_build_response_tables(raw),
        citations=_build_citations(raw),
        approval_form=_build_approval_form(raw),
        booking_draft=_build_booking_draft(raw),
        execution_plan=metadata.get("execution_plan"),
        policy_constraints=metadata.get("policy_constraints"),
        policy_validation=metadata.get("policy_validation"),
        travel_retry=metadata.get("travel_retry"),
        trace=metadata.get("trace") or [],
        risk_level=metadata.get("risk_level"),
        answer_mode=metadata.get("answer_mode"),
        verification=metadata.get("verification"),
    )
