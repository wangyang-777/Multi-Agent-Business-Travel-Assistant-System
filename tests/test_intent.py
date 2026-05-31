"""Unit tests for intent recognizer."""

from __future__ import annotations

import pytest

from app.core.intent.recognizer import IntentRecognizer, TravelIntent


@pytest.mark.asyncio
async def test_recognize_flight_intent() -> None:
    r = IntentRecognizer()
    out = await r.recognize("帮我查一下明天北京到上海的机票")
    assert out.intent is TravelIntent.SEARCH_FLIGHT
    assert out.confidence >= 0.4


@pytest.mark.asyncio
async def test_recognize_policy_intent() -> None:
    r = IntentRecognizer()
    out = await r.recognize("公司的差标报销政策上限是多少")
    assert out.intent is TravelIntent.POLICY


@pytest.mark.asyncio
async def test_recognize_general_fallback() -> None:
    r = IntentRecognizer()
    out = await r.recognize("你好")
    assert out.intent is TravelIntent.GENERAL
