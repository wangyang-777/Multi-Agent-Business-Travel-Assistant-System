"""Travel inventory search and recommendation tools."""

from __future__ import annotations

import time
import asyncio
import json
import os
from datetime import date, datetime, time as dt_time, timedelta
from decimal import Decimal
from enum import Enum
import re
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, Field

from app.config import settings
from app.core.tools.mcp_client import MCPHttpClient


class TravelMode(str, Enum):
    FLIGHT = "flight"
    HOTEL = "hotel"
    TRAIN = "train"


class FlightSearchRequest(BaseModel):
    origin: str = Field(..., description="IATA code or city name")
    destination: str
    depart_date: date
    return_date: date | None = None
    cabin: str = "economy"
    passengers: int = Field(1, ge=1, le=9)


class HotelSearchRequest(BaseModel):
    city: str
    check_in: date
    check_out: date
    keyword: str | None = None
    poi_name: str | None = Field(default=None, description="Nearby POI/business district/customer site")
    guests: int = Field(1, ge=1, le=9)
    max_nightly_cny: Decimal | None = None


class TrainSearchRequest(BaseModel):
    origin_station: str
    dest_station: str
    depart_date: date
    prefer_gd: bool = True
    limited_num: int = Field(10, ge=1, le=20)


class TravelRecommendationRequest(BaseModel):
    employee_id: str
    grade: str
    origin_city: str
    destination_city: str
    departure_date: date
    return_date: date | None = None
    purpose: str = "client_visit"
    preferred_class: str = "economy"
    hotel_budget_cny: Decimal | None = None
    passenger_count: int = Field(1, ge=1, le=9)
    hotel_keyword: str | None = None
    hotel_nearby_poi: str | None = None
    include_trains: bool = True
    prefer_gd_trains: bool = True


class TravelInventoryProvider(Protocol):
    async def search_flights(self, req: FlightSearchRequest) -> list[dict[str, Any]]:
        ...

    async def search_hotels(self, req: HotelSearchRequest) -> list[dict[str, Any]]:
        ...


_CITY_CODES = {
    "北京": "BJS",
    "上海": "SHA",
    "广州": "CAN",
    "深圳": "SZX",
    "杭州": "HGH",
    "成都": "CTU",
    "重庆": "CKG",
    "南京": "NKG",
    "武汉": "WUH",
    "西安": "SIA",
    "济南": "TNA",
    "青岛": "TAO",
}

_MCP_STATION_TOOL_CANDIDATES = (
    ("get-station-code-of-citys", "citys"),
    ("get-station-code-of-city", "city"),
    ("get-station-code-by-names", "stationNames"),
    ("get-station-code-by-name", "stationName"),
)


def _city_code(value: str) -> str:
    clean = value.strip().upper()
    if len(clean) == 3 and clean.isalpha():
        return clean
    return _CITY_CODES.get(value.strip(), clean[:3])


def _fmt_dt(day: date, hour: int, minute: int = 0) -> str:
    return datetime.combine(day, dt_time(hour, minute)).isoformat()


def _today_china() -> date:
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


def _mcp_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, list):
            parts = [
                str(item.get("text"))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
            ]
            return "\n".join(parts)
        if "text" in payload:
            return str(payload["text"])
        return json_dumps(payload)
    return json_dumps(payload)


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _money_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if re.search(r"\d+\s*[xX]", text):
        return text
    match = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    return match.group(1) if match else text


def _money_decimal(value: Any) -> Decimal | None:
    text = _money_text(value)
    if re.search(r"[xX]", text):
        return None
    try:
        return Decimal(text) if text else None
    except Exception:
        return None


def _extract_flyai_items(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("itemList", "items", "list", "resultList"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    for key in ("itemList", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _build_12306_skill_command(req: TrainSearchRequest) -> list[str]:
    base_dir = Path(settings.railway_12306_skill_dir)
    cmd = [
        settings.railway_12306_node_bin,
        str(base_dir / "scripts" / "query.mjs"),
        req.origin_station,
        req.dest_station,
        "-d",
        req.depart_date.isoformat(),
        "--json",
    ]
    if req.prefer_gd:
        cmd.extend(["-t", "GD"])
    return cmd


def _extract_result_list(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("results", "trains", "tickets", "data", "rows"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _first_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _seat_available(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text or text in {"--", "-", "无", "候补", "不可用", "不适用"}:
        return False
    return bool(re.search(r"\d+", text)) or text in {"有", "充足", "少量", "Y", "y", "yes", "true"}


def _parse_hhmm(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    match = re.search(r"(\d{1,2}):(\d{2})", str(value))
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def _minutes_from_hhmm(value: Any) -> int | None:
    parsed = _parse_hhmm(value)
    if parsed is None:
        return None
    return parsed[0] * 60 + parsed[1]


def _duration_minutes(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(?:(\d+)\s*(?:小时|h|H))?\s*(?:(\d+)\s*(?:分钟|m|M))?", text)
    if match and (match.group(1) or match.group(2)):
        return int(match.group(1) or 0) * 60 + int(match.group(2) or 0)
    match = re.search(r"(\d{1,2}):(\d{2})", text)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


def _train_code(row: dict[str, Any]) -> str:
    return str(_first_value(row, "trainCode", "train_code", "trainNo", "train_no", "station_train_code") or "")


def _train_depart_time(row: dict[str, Any]) -> Any:
    return _first_value(row, "departTime", "depart_time", "startTime", "start_time", "start_time_text")


def _train_arrive_time(row: dict[str, Any]) -> Any:
    return _first_value(row, "arriveTime", "arrive_time", "arrivalTime", "arrival_time", "arrive_time_text")


def _train_duration(row: dict[str, Any]) -> Any:
    return _first_value(row, "duration", "lishi", "elapsedTime", "elapsed_time", "travel_time")


def _train_can_buy(row: dict[str, Any]) -> bool:
    value = _first_value(row, "canBuy", "can_buy", "canWebBuy", "can_web_buy", "isBookable", "available")
    if isinstance(value, bool):
        return value
    text = str(value or "").strip()
    if text in {"Y", "y", "yes", "true", "True", "可购", "可预订"}:
        return True
    if text in {"N", "n", "no", "false", "False", "不可购", "停运"}:
        return False
    return _seat_available(_first_value(row, "ze", "zy", "swz", "tz", "yw", "rw", "dw", "yz", "wz"))


def _train_recommendation(row: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    code = _train_code(row).upper()

    if _train_can_buy(row):
        score += 40
        reasons.append("当前显示可购")
    else:
        score -= 60
        reasons.append("当前未显示可购，降低推荐优先级")

    if _seat_available(_first_value(row, "ze", "second_class", "secondClass")):
        score += 28
        reasons.append("二等座有票，符合 staff 常规差旅标准")
    elif _seat_available(_first_value(row, "zy", "first_class", "firstClass")):
        score += 12
        reasons.append("一等座有票，可作为备选但需关注差标")
    elif _seat_available(_first_value(row, "swz", "tz", "business_class", "special_class")):
        score += 4
        reasons.append("高等级席别有票，通常需要额外审批")
    else:
        reasons.append("未识别到合适席别余票")

    if code.startswith(("G", "D")):
        score += 14
        reasons.append("G/D 字头，适合商务出行效率")
    elif code.startswith("C"):
        score += 8
        reasons.append("城际列车，效率较高")

    depart_min = _minutes_from_hhmm(_train_depart_time(row))
    if depart_min is not None:
        if 7 * 60 <= depart_min <= 10 * 60:
            score += 22
            reasons.append("早高峰后至上午出发，便于当天到达办事")
        elif 6 * 60 <= depart_min < 7 * 60:
            score += 12
            reasons.append("较早出发，适合赶上午行程")
        elif 10 * 60 < depart_min <= 12 * 60:
            score += 10
            reasons.append("上午出发，时间仍较可控")
        elif 12 * 60 < depart_min <= 18 * 60:
            score += 3
            reasons.append("下午出发，适合非上午会议")
        else:
            score -= 8
            reasons.append("出发时间偏早或偏晚，商务舒适度较低")

    arrive_min = _minutes_from_hhmm(_train_arrive_time(row))
    if arrive_min is not None:
        if arrive_min <= 12 * 60:
            score += 10
            reasons.append("中午前到达，适合当天开展工作")
        elif arrive_min <= 18 * 60:
            score += 6
            reasons.append("傍晚前到达，仍适合商务安排")
        else:
            score -= 6
            reasons.append("到达时间偏晚")

    duration_min = _duration_minutes(_train_duration(row))
    if duration_min is not None:
        if duration_min <= 5 * 60:
            score += 20
            reasons.append("全程耗时短")
        elif duration_min <= 6 * 60:
            score += 14
            reasons.append("全程耗时可接受")
        elif duration_min <= 8 * 60:
            score += 6
            reasons.append("耗时略长")
        else:
            score -= 8
            reasons.append("耗时较长")

    return score, reasons[:5]


def _rank_train_results(results: list[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for index, row in enumerate(results):
        enriched = dict(row)
        score, reasons = _train_recommendation(enriched)
        enriched["recommendation_score"] = score
        enriched["recommendation_reasons"] = reasons
        enriched["reason"] = "；".join(reasons)
        enriched["_original_index"] = index
        ranked.append(enriched)
    ranked.sort(
        key=lambda item: (
            -int(item.get("recommendation_score") or 0),
            _minutes_from_hhmm(_train_depart_time(item)) if _minutes_from_hhmm(_train_depart_time(item)) is not None else 24 * 60,
            _duration_minutes(_train_duration(item)) if _duration_minutes(_train_duration(item)) is not None else 10**6,
            int(item.get("_original_index") or 0),
        )
    )
    for row in ranked:
        row.pop("_original_index", None)
    return ranked[:limit] if limit is not None else ranked


def _normalize_12306_skill_results(payload: Any, *, limit: int) -> list[dict[str, Any]]:
    rows = _extract_result_list(payload)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append(row)
        else:
            normalized.append({"raw": row})
    return _rank_train_results(normalized, limit=limit)


async def _search_trains_via_12306_skill(req: TrainSearchRequest) -> dict[str, Any]:
    if not settings.railway_12306_skill_dir:
        return {
            "mode": TravelMode.TRAIN.value,
            "provider": "12306_skill",
            "query": req.model_dump(mode="json"),
            "results": [],
            "raw_text": "",
            "disclaimer": "未配置 RAILWAY_12306_SKILL_DIR，未执行本地 12306 Skill 查询；系统不会生成演示车票。",
            "error": "railway_12306_skill_dir_not_configured",
        }

    script = Path(settings.railway_12306_skill_dir) / "scripts" / "query.mjs"
    if not script.exists():
        return {
            "mode": TravelMode.TRAIN.value,
            "provider": "12306_skill",
            "query": req.model_dump(mode="json"),
            "results": [],
            "raw_text": "",
            "disclaimer": "已配置 RAILWAY_12306_SKILL_DIR，但未找到 scripts/query.mjs；未生成演示车票。",
            "error": f"query_script_not_found: {script}",
        }

    cmd = _build_12306_skill_command(req)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=settings.railway_mcp_timeout_s,
        )
    except FileNotFoundError as exc:
        return {
            "mode": TravelMode.TRAIN.value,
            "provider": "12306_skill",
            "query": req.model_dump(mode="json"),
            "results": [],
            "raw_text": "",
            "disclaimer": "本地 12306 Skill 需要 Node.js，但当前环境未找到 node；未生成演示车票。",
            "error": str(exc),
        }
    except TimeoutError:
        return {
            "mode": TravelMode.TRAIN.value,
            "provider": "12306_skill",
            "query": req.model_dump(mode="json"),
            "results": [],
            "raw_text": "",
            "disclaimer": "本地 12306 Skill 查询超时；未生成演示车票。",
            "error": "query_timeout",
        }

    out_text = stdout.decode("utf-8", errors="ignore").strip()
    err_text = stderr.decode("utf-8", errors="ignore").strip()
    if proc.returncode != 0:
        return {
            "mode": TravelMode.TRAIN.value,
            "provider": "12306_skill",
            "query": req.model_dump(mode="json"),
            "results": [],
            "raw_text": out_text,
            "disclaimer": "本地 12306 Skill 查询失败；未生成演示车票。",
            "error": err_text or f"node exited with {proc.returncode}",
        }

    try:
        payload = json.loads(out_text) if out_text else []
    except json.JSONDecodeError:
        payload = {"raw_text": out_text}
    results = _normalize_12306_skill_results(payload, limit=req.limited_num)
    raw_text = json_dumps(payload)
    return {
        "mode": TravelMode.TRAIN.value,
        "provider": "12306_skill",
        "query": req.model_dump(mode="json"),
        "results": results,
        "raw_text": raw_text,
        "stderr": err_text,
        "disclaimer": "车票信息来自本地 12306 Skill 查询结果；正式预订前仍需以 12306 官方页面库存、票价和席别为准。",
    }


def _extract_station_code(text: str, preferred_name: str = "") -> str | None:
    if not text:
        return None
    if preferred_name:
        pattern = rf"{re.escape(preferred_name)}[^\nA-Z]{{0,20}}([A-Z]{{3}})"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    candidates = re.findall(r"\b[A-Z]{3}\b", text)
    return candidates[0] if candidates else None


async def _mcp_call_tool(client: MCPHttpClient, names: tuple[str, ...], arguments: dict[str, Any]) -> tuple[str, Any]:
    last_error: Exception | None = None
    for name in names:
        try:
            return name, await client.call_tool(name, arguments)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    if last_error:
        raise last_error
    raise RuntimeError("no MCP tool candidates provided")


async def _lookup_station_code(client: MCPHttpClient, station_or_city: str) -> tuple[str | None, str]:
    errors: list[str] = []
    for tool_name, arg_name in _MCP_STATION_TOOL_CANDIDATES:
        value = station_or_city
        try:
            payload = await client.call_tool(tool_name, {arg_name: value})
            text = _mcp_text(payload)
            code = _extract_station_code(text, station_or_city)
            if code:
                return code, text
            errors.append(f"{tool_name}: 未解析到 station_code")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{tool_name}: {exc}")
    return None, "；".join(errors)


async def _search_trains_via_12306_mcp(req: TrainSearchRequest) -> dict[str, Any]:
    if not settings.railway_mcp_url:
        return {
            "mode": TravelMode.TRAIN.value,
            "provider": "12306_mcp",
            "query": req.model_dump(mode="json"),
            "results": [],
            "raw_text": "",
            "disclaimer": "未配置 RAILWAY_MCP_URL，未执行真实 12306 余票查询；系统不会生成演示车票。",
            "error": "railway_mcp_url_not_configured",
        }

    client = MCPHttpClient(settings.railway_mcp_url, timeout_s=settings.railway_mcp_timeout_s)
    try:
        try:
            await client.initialize("travel-agent-guide")
        except Exception:
            pass
        from_code, from_debug = await _lookup_station_code(client, req.origin_station)
        to_code, to_debug = await _lookup_station_code(client, req.dest_station)
        if not from_code or not to_code:
            return {
                "mode": TravelMode.TRAIN.value,
                "provider": "12306_mcp",
                "query": req.model_dump(mode="json"),
                "results": [],
                "raw_text": "",
                "disclaimer": "已尝试调用 12306 MCP，但未能解析出出发地或到达地 station_code；未生成演示车票。",
                "error": "station_code_lookup_failed",
                "debug": {"from": from_debug, "to": to_debug},
            }

        train_filter_flags = "GDC" if req.prefer_gd else ""
        _, payload = await _mcp_call_tool(
            client,
            ("get-tickets",),
            {
                "date": req.depart_date.isoformat(),
                "fromStation": from_code,
                "toStation": to_code,
                "trainFilterFlags": train_filter_flags,
                "limitedNum": req.limited_num,
            },
        )
        raw_text = _mcp_text(payload)
        return {
            "mode": TravelMode.TRAIN.value,
            "provider": "12306_mcp",
            "query": {
                **req.model_dump(mode="json"),
                "from_station_code": from_code,
                "to_station_code": to_code,
            },
            "results": [{"raw_text": raw_text}] if raw_text else [],
            "raw_text": raw_text,
            "disclaimer": "车票信息来自 12306 MCP 实时查询结果；正式预订前仍需以 12306 官方页面库存、票价和席别为准。",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "mode": TravelMode.TRAIN.value,
            "provider": "12306_mcp",
            "query": req.model_dump(mode="json"),
            "results": [],
            "raw_text": "",
            "disclaimer": "12306 MCP 查询失败，未生成演示车票；请稍后重试或人工查询 12306。",
            "error": str(exc),
        }
    finally:
        await client.close()


class DemoTravelInventoryProvider:
    async def search_flights(self, req: FlightSearchRequest) -> list[dict[str, Any]]:
        direct_price = Decimal("1280")
        if req.origin != req.destination:
            direct_price += Decimal("220")
        cabin_factor = Decimal("2.4") if req.cabin == "business" else Decimal("1")
        base = (direct_price * cabin_factor).quantize(Decimal("0.01"))
        return [
            {
                "provider": "demo",
                "carrier": "示例航司",
                "flight_no": "CA1501",
                "origin": req.origin,
                "destination": req.destination,
                "depart_at": _fmt_dt(req.depart_date, 8, 30),
                "arrive_at": _fmt_dt(req.depart_date, 10, 55),
                "duration": "2h25m",
                "cabin": req.cabin,
                "price_cny": str(base),
                "refundable": False,
                "reason": "早到达，便于当天开展会议。",
            },
            {
                "provider": "demo",
                "carrier": "示例航司",
                "flight_no": "MU5102",
                "origin": req.origin,
                "destination": req.destination,
                "depart_at": _fmt_dt(req.depart_date, 10, 15),
                "arrive_at": _fmt_dt(req.depart_date, 12, 35),
                "duration": "2h20m",
                "cabin": req.cabin,
                "price_cny": str((base - Decimal("160")).quantize(Decimal("0.01"))),
                "refundable": True,
                "reason": "价格更低，时间仍适合上午出发。",
            },
        ][: settings.travel_search_max_results]

    async def search_hotels(self, req: HotelSearchRequest) -> list[dict[str, Any]]:
        nights = max((req.check_out - req.check_in).days, 1)
        candidates = [
            (f"{req.city}商务精选酒店", 4, Decimal("560"), "距核心商务区约 2km"),
            (f"{req.city}中心智选酒店", 4, Decimal("720"), "交通方便，适合客户拜访"),
            (f"{req.city}行政公寓", 5, Decimal("920"), "房型更舒适，但可能超过 staff 标准"),
        ]
        location_hint = f"；位置约束：{req.poi_name}附近" if req.poi_name else ""
        out: list[dict[str, Any]] = []
        for name, star, nightly, reason in candidates:
            if req.max_nightly_cny is not None and nightly > req.max_nightly_cny:
                continue
            out.append(
                {
                    "provider": "demo",
                    "name": name,
                    "city": req.city,
                    "star": star,
                    "check_in": req.check_in.isoformat(),
                    "check_out": req.check_out.isoformat(),
                    "nights": nights,
                    "nightly_cny": str(nightly),
                    "total_cny": str((nightly * nights).quantize(Decimal("0.01"))),
                    "poi_name": req.poi_name,
                    "reason": f"{reason}{location_hint}",
                }
            )
        return out[: settings.travel_search_max_results]


class AmadeusTravelInventoryProvider:
    def __init__(self) -> None:
        self._token: str | None = None
        self._token_expires_at = 0.0

    async def _access_token(self) -> str:
        now = time.time()
        if self._token and now < self._token_expires_at - 60:
            return self._token
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{settings.amadeus_base_url}/v1/security/oauth2/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": settings.amadeus_client_id,
                    "client_secret": settings.amadeus_client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            payload = resp.json()
        self._token = str(payload["access_token"])
        self._token_expires_at = now + int(payload.get("expires_in", 1799))
        return self._token

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        token = await self._access_token()
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{settings.amadeus_base_url}{path}",
                params={k: v for k, v in params.items() if v not in (None, "")},
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            return resp.json()

    async def search_flights(self, req: FlightSearchRequest) -> list[dict[str, Any]]:
        payload = await self._get(
            "/v2/shopping/flight-offers",
            {
                "originLocationCode": _city_code(req.origin),
                "destinationLocationCode": _city_code(req.destination),
                "departureDate": req.depart_date.isoformat(),
                "returnDate": req.return_date.isoformat() if req.return_date else None,
                "adults": req.passengers,
                "travelClass": req.cabin.upper() if req.cabin else None,
                "currencyCode": "CNY",
                "max": settings.travel_search_max_results,
            },
        )
        out: list[dict[str, Any]] = []
        for offer in payload.get("data", []):
            first_segment = (
                offer.get("itineraries", [{}])[0].get("segments", [{}])[0]
                if offer.get("itineraries")
                else {}
            )
            price = offer.get("price") or {}
            carrier = ",".join(offer.get("validatingAirlineCodes") or [])
            out.append(
                {
                    "provider": "amadeus",
                    "carrier": carrier or first_segment.get("carrierCode"),
                    "flight_no": f"{first_segment.get('carrierCode', '')}{first_segment.get('number', '')}",
                    "origin": first_segment.get("departure", {}).get("iataCode"),
                    "destination": first_segment.get("arrival", {}).get("iataCode"),
                    "depart_at": first_segment.get("departure", {}).get("at"),
                    "arrive_at": first_segment.get("arrival", {}).get("at"),
                    "duration": offer.get("itineraries", [{}])[0].get("duration"),
                    "cabin": req.cabin,
                    "price_cny": price.get("grandTotal") or price.get("total"),
                    "currency": price.get("currency", "CNY"),
                    "refundable": None,
                    "reason": "供应商实时返回的可售航班。",
                }
            )
        return out

    async def search_hotels(self, req: HotelSearchRequest) -> list[dict[str, Any]]:
        hotels = await self._get(
            "/v1/reference-data/locations/hotels/by-city",
            {"cityCode": _city_code(req.city), "radius": 8, "radiusUnit": "KM"},
        )
        hotel_ids = [
            item.get("hotelId")
            for item in hotels.get("data", [])
            if item.get("hotelId")
        ][:20]
        if not hotel_ids:
            return []
        offers = await self._get(
            "/v3/shopping/hotel-offers",
            {
                "hotelIds": ",".join(hotel_ids),
                "adults": req.guests,
                "checkInDate": req.check_in.isoformat(),
                "checkOutDate": req.check_out.isoformat(),
                "currency": "CNY",
                "bestRateOnly": "true",
            },
        )
        out: list[dict[str, Any]] = []
        nights = max((req.check_out - req.check_in).days, 1)
        for item in offers.get("data", []):
            hotel = item.get("hotel") or {}
            offer = (item.get("offers") or [{}])[0]
            price = offer.get("price") or {}
            total = Decimal(str(price.get("total") or "0"))
            nightly = (total / Decimal(nights)).quantize(Decimal("0.01")) if nights else total
            if req.max_nightly_cny is not None and nightly > req.max_nightly_cny:
                continue
            out.append(
                {
                    "provider": "amadeus",
                    "name": hotel.get("name"),
                    "city": req.city,
                    "star": hotel.get("rating"),
                    "check_in": req.check_in.isoformat(),
                    "check_out": req.check_out.isoformat(),
                    "nights": nights,
                    "nightly_cny": str(nightly),
                    "total_cny": str(total),
                    "currency": price.get("currency", "CNY"),
                    "reason": "供应商实时返回的可售酒店报价。",
                }
            )
        return out[: settings.travel_search_max_results]


class FlyAITravelInventoryProvider:
    async def _run(self, args: list[str]) -> dict[str, Any]:
        env = os.environ.copy()
        if settings.flyai_api_key:
            env["FLYAI_API_KEY"] = settings.flyai_api_key
        try:
            proc = await asyncio.create_subprocess_exec(
                settings.flyai_cli_bin,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=settings.flyai_timeout_s)
        except FileNotFoundError as exc:
            raise RuntimeError("FlyAI CLI 未安装或不可执行，请安装 @fly-ai/flyai-cli。") from exc
        except TimeoutError as exc:
            raise RuntimeError("FlyAI CLI 查询超时。") from exc

        out_text = stdout.decode("utf-8", errors="ignore").strip()
        err_text = stderr.decode("utf-8", errors="ignore").strip()
        if proc.returncode != 0:
            raise RuntimeError(err_text or f"FlyAI CLI exited with {proc.returncode}")
        try:
            payload = json.loads(out_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"FlyAI CLI 返回非 JSON 内容：{out_text[:300]}") from exc
        if isinstance(payload, dict) and payload.get("status") not in (None, 0, "0"):
            raise RuntimeError(str(payload.get("message") or payload.get("systemMessage") or "FlyAI 查询失败"))
        return payload if isinstance(payload, dict) else {"data": payload}

    async def search_flights(self, req: FlightSearchRequest) -> list[dict[str, Any]]:
        args = [
            "search-flight",
            "--origin",
            req.origin,
            "--destination",
            req.destination,
            "--dep-date",
            req.depart_date.isoformat(),
            "--sort-type",
            "2",
        ]
        if req.return_date:
            args.extend(["--back-date", req.return_date.isoformat()])
        if req.cabin:
            cabin_map = {
                "economy": "经济舱",
                "business": "商务舱",
                "first": "头等舱",
            }
            args.extend(["--seat-class-name", cabin_map.get(req.cabin, req.cabin)])

        payload = await self._run(args)
        out: list[dict[str, Any]] = []
        for item in _extract_flyai_items(payload)[: settings.travel_search_max_results]:
            journey = (item.get("journeys") or [{}])[0] if isinstance(item.get("journeys"), list) else {}
            segment = (journey.get("segments") or [{}])[0] if isinstance(journey.get("segments"), list) else {}
            out.append(
                {
                    "provider": "flyai",
                    "carrier": segment.get("marketingTransportName"),
                    "flight_no": segment.get("marketingTransportNo") or item.get("transportNo"),
                    "origin": segment.get("depCityName") or req.origin,
                    "destination": segment.get("arrCityName") or req.destination,
                    "depart_at": segment.get("depDateTime"),
                    "arrive_at": segment.get("arrDateTime"),
                    "duration": item.get("totalDuration") or journey.get("totalDuration") or segment.get("duration"),
                    "cabin": segment.get("seatClassName") or req.cabin,
                    "price_cny": _money_text(item.get("adultPrice") or item.get("ticketPrice") or item.get("price")),
                    "currency": "CNY",
                    "refundable": None,
                    "booking_url": item.get("jumpUrl"),
                    "system_message": payload.get("systemMessage"),
                    "reason": "FlyAI/飞猪实时返回的航班候选。",
                }
            )
        return out

    async def search_hotels(self, req: HotelSearchRequest) -> list[dict[str, Any]]:
        args = [
            "search-hotel",
            "--dest-name",
            req.city,
            "--check-in-date",
            req.check_in.isoformat(),
            "--check-out-date",
            req.check_out.isoformat(),
            "--sort",
            "price_asc",
        ]
        if req.keyword:
            args.extend(["--key-words", req.keyword])
        if req.poi_name:
            args.extend(["--poi-name", req.poi_name])
        if req.max_nightly_cny is not None:
            args.extend(["--max-price", str(int(req.max_nightly_cny))])

        payload = await self._run(args)
        nights = max((req.check_out - req.check_in).days, 1)
        out: list[dict[str, Any]] = []
        for item in _extract_flyai_items(payload)[: settings.travel_search_max_results]:
            nightly = _money_decimal(item.get("price"))
            if req.max_nightly_cny is not None and nightly is not None and nightly > req.max_nightly_cny:
                continue
            out.append(
                {
                    "provider": "flyai",
                    "name": item.get("name"),
                    "city": req.city,
                    "star": item.get("star"),
                    "address": item.get("address"),
                    "poi_name": req.poi_name,
                    "nearby": item.get("interestsPoi"),
                    "score": item.get("score"),
                    "score_desc": item.get("scoreDesc"),
                    "check_in": req.check_in.isoformat(),
                    "check_out": req.check_out.isoformat(),
                    "nights": nights,
                    "nightly_cny": str(nightly) if nightly is not None else _money_text(item.get("price")),
                    "total_cny": str((nightly * nights).quantize(Decimal("0.01"))) if nightly is not None else "",
                    "currency": "CNY",
                    "main_pic": item.get("mainPic"),
                    "detail_url": item.get("detailUrl"),
                    "system_message": payload.get("systemMessage"),
                    "reason": item.get("review") or item.get("interestsPoi") or "FlyAI/飞猪实时返回的酒店候选。",
                }
            )
        return out[: settings.travel_search_max_results]


def _provider() -> TravelInventoryProvider:
    if settings.travel_inventory_provider.lower() == "flyai":
        return FlyAITravelInventoryProvider()
    if (
        settings.travel_inventory_provider.lower() == "amadeus"
        and settings.amadeus_client_id
        and settings.amadeus_client_secret
    ):
        return AmadeusTravelInventoryProvider()
    return DemoTravelInventoryProvider()


async def search_flights(req: FlightSearchRequest) -> dict[str, Any]:
    provider = _provider()
    try:
        results = await provider.search_flights(req)
    except Exception as exc:  # noqa: BLE001
        provider_name = settings.travel_inventory_provider.lower()
        return {
            "mode": TravelMode.FLIGHT.value,
            "provider": provider_name,
            "query": req.model_dump(mode="json"),
            "results": [],
            "error": str(exc),
            "disclaimer": f"{provider_name} 航班查询失败，系统未生成演示航班；请稍后重试或人工确认。",
        }
    provider_name = results[0]["provider"] if results else settings.travel_inventory_provider
    return {
        "mode": TravelMode.FLIGHT.value,
        "provider": provider_name,
        "query": req.model_dump(mode="json"),
        "results": results,
        "disclaimer": (
            "价格与库存需在预订前再次确认；demo provider 返回演示数据。"
            if provider_name == "demo"
            else "价格与库存来自供应商接口返回，正式预订前仍需再次确认退改规则和审批状态。"
        ),
    }


async def search_hotels(req: HotelSearchRequest) -> dict[str, Any]:
    provider = _provider()
    try:
        results = await provider.search_hotels(req)
    except Exception as exc:  # noqa: BLE001
        provider_name = settings.travel_inventory_provider.lower()
        return {
            "mode": TravelMode.HOTEL.value,
            "provider": provider_name,
            "query": req.model_dump(mode="json"),
            "results": [],
            "error": str(exc),
            "disclaimer": f"{provider_name} 酒店查询失败，系统未生成演示酒店；请稍后重试或人工确认。",
        }
    provider_name = results[0]["provider"] if results else settings.travel_inventory_provider
    return {
        "mode": TravelMode.HOTEL.value,
        "provider": provider_name,
        "query": req.model_dump(mode="json"),
        "results": results,
        "disclaimer": (
            "价格与库存需在预订前再次确认；demo provider 返回演示数据。"
            if provider_name == "demo"
            else "价格与库存来自供应商接口返回，正式预订前仍需再次确认退改规则和审批状态。"
        ),
    }


async def recommend_travel_options(req: TravelRecommendationRequest) -> dict[str, Any]:
    checkout = req.return_date or (req.departure_date + timedelta(days=1))
    days_before_departure = (req.departure_date - _today_china()).days
    advance_booking_days_min = 7
    policy_warnings: list[str] = []
    if days_before_departure < advance_booking_days_min:
        policy_warnings.append(
            f"距出发仅 {days_before_departure} 天，低于提前 {advance_booking_days_min} 天预订要求，"
            "可能产生附加费并需要说明原因。"
        )
    flight_task = search_flights(
        FlightSearchRequest(
            origin=req.origin_city,
            destination=req.destination_city,
            depart_date=req.departure_date,
            return_date=req.return_date,
            cabin=req.preferred_class,
            passengers=req.passenger_count,
        )
    )
    hotel_task = search_hotels(
        HotelSearchRequest(
            city=req.destination_city,
            check_in=req.departure_date,
            check_out=checkout,
            keyword=req.hotel_keyword,
            poi_name=req.hotel_nearby_poi,
            guests=req.passenger_count,
            max_nightly_cny=req.hotel_budget_cny,
        )
    )
    train_task = (
        search_trains(
            TrainSearchRequest(
                origin_station=req.origin_city,
                dest_station=req.destination_city,
                depart_date=req.departure_date,
                prefer_gd=req.prefer_gd_trains,
                limited_num=max(1, min(settings.travel_search_max_results, 20)),
            )
        )
        if req.include_trains
        else None
    )

    if train_task is not None:
        flights, hotels, trains = await asyncio.gather(flight_task, hotel_task, train_task)
    else:
        flights, hotels = await asyncio.gather(flight_task, hotel_task)
        trains = {
            "mode": TravelMode.TRAIN.value,
            "provider": "disabled",
            "query": {},
            "results": [],
            "disclaimer": "本次推荐未启用高铁/火车候选查询。",
        }

    ranked_trains = _rank_train_results([r for r in trains.get("results", []) if isinstance(r, dict)])
    best_flight = (flights.get("results") or [None])[0]
    best_train = (ranked_trains or [None])[0]
    best_hotel = (hotels.get("results") or [None])[0]
    providers = {
        "flights": flights.get("provider"),
        "trains": trains.get("provider"),
        "hotels": hotels.get("provider"),
    }
    has_demo_inventory = flights.get("provider") == "demo" or hotels.get("provider") == "demo"
    has_train_error = bool(trains.get("error"))
    return {
        "mode": "travel_recommendation",
        "provider": flights.get("provider") or hotels.get("provider") or trains.get("provider") or settings.travel_inventory_provider,
        "providers": providers,
        "query": req.model_dump(mode="json"),
        "flights": flights.get("results", []),
        "trains": ranked_trains,
        "hotels": hotels.get("results", []),
        "policy_checks": {
            "days_before_departure": days_before_departure,
            "advance_booking_days_min": advance_booking_days_min,
            "advance_booking_ok": days_before_departure >= advance_booking_days_min,
        },
        "policy_warnings": policy_warnings,
        "recommendation": {
            "flight": best_flight,
            "train": best_train,
            "hotel": best_hotel,
            "reason": (
                "综合比较航班、高铁/火车和酒店候选；优先选择时间适合商务行程、"
                "价格在差标内、库存状态明确且可解释性较好的组合。"
            ),
        },
        "disclaimer": (
            "当前航班/酒店候选来自 demo provider，属于演示数据，非实时库存或最终价格；"
            "高铁/火车候选来自 12306 查询结果或明确的失败状态；正式预订前需接入真实供应商"
            "或人工确认价格、库存、退改规则和公司审批状态。"
            if has_demo_inventory
            else (
                "高铁/火车候选查询失败或未配置，系统未生成演示车票；当前结果用于推荐与比选，"
                "正式预订前需再次确认价格、库存、退改规则和公司审批状态。"
                if has_train_error
                else "当前结果用于推荐与比选，正式预订前需再次确认价格、库存、退改规则和公司审批状态。"
            )
        ),
    }


async def search_trains(req: TrainSearchRequest) -> dict[str, Any]:
    if settings.railway_12306_skill_dir:
        return await _search_trains_via_12306_skill(req)
    if settings.railway_mcp_url:
        return await _search_trains_via_12306_mcp(req)
    return {
        "mode": TravelMode.TRAIN.value,
        "provider": "railway_unconfigured",
        "query": req.model_dump(mode="json"),
        "results": [],
        "raw_text": "",
        "disclaimer": (
            "未配置 RAILWAY_12306_SKILL_DIR 或 RAILWAY_MCP_URL，未执行真实 12306 余票查询；"
            "系统不会生成演示车票。"
        ),
        "error": "railway_provider_not_configured",
    }
