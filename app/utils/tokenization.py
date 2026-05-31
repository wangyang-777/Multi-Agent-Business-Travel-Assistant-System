from __future__ import annotations

from functools import lru_cache
import re

import tiktoken


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding | None:
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _fallback_token_count(text: str) -> int:
    # Conservative offline fallback: CJK chars and punctuation count directly;
    # ASCII words are grouped for count, but truncation below remains char-safe.
    return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", text or ""))


def count_text_tokens(text: str) -> int:
    enc = _encoding()
    if enc is None:
        return _fallback_token_count(text)
    return len(enc.encode(text or ""))


def truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    enc = _encoding()
    if enc is None:
        kept: list[str] = []
        used = 0
        for char in text or "":
            token_count = _fallback_token_count(char)
            if used + token_count > max_tokens:
                break
            kept.append(char)
            used += token_count
        return "".join(kept).strip()

    tokens = enc.encode(text or "")
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens]).strip()
