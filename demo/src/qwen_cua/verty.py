"""Optional Verty-CUA guard: a cheap visual check between chained actions.

The model may return several tool calls in one turn -- click, then type, then
Enter. The runner executes that chain blindly, which is only sound while each
intermediate state is what the model assumed when it planned the chain. When a
click lands on nothing, or opens a dialog the model did not anticipate, every
remaining action in the chain is aimed at a screen that no longer exists.

`verty-serve` answers, in ~10 ms and with no model, whether the screen did what
each action should have done. Guard mode uses that to stop a chain the moment it
stops making sense, and hands control back to the model a turn earlier than it
would otherwise notice.

This is deliberately the *conservative* half of the idea. It spends no fewer
model turns than the unguarded runner -- it may spend one more -- and buys
correctness instead. The turn-skipping half (acting on `expected` to avoid a
model call entirely) trades the other way and is not implemented here.

Disabled unless QWEN_CUA_VERTY_URL is set, and a failure to reach the service is
never fatal: the guard degrades to the unguarded behaviour rather than taking
the run down with it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx


@dataclass(frozen=True, slots=True)
class VertySettings:
    url: str = ""
    guard: bool = True
    # Fold verified frames to text in later turns instead of shipping them as
    # images. This is the half that trades the other way from guarding: it saves
    # vision tokens on evidence the model has already accounted for, and is the
    # only thing here that can actually reduce cost.
    fold: bool = False

    # Consecutive steps introducing no new content before the chain is stopped.
    # A repaint the screen has shown before -- a focus ring, a caret, a hover --
    # is change, but it is not progress; this counts the latter.
    stuck_run: int = 3
    timeout_s: float = 5.0

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    @classmethod
    def from_env(cls) -> VertySettings:
        return cls(
            url=os.getenv("QWEN_CUA_VERTY_URL", "").strip().rstrip("/"),
            guard=os.getenv("QWEN_CUA_VERTY_GUARD", "true").strip().lower()
            not in {"0", "false", "no"},
            fold=os.getenv("QWEN_CUA_VERTY_FOLD", "false").strip().lower()
            in {"1", "true", "yes"},
            stuck_run=max(0, int(os.getenv("QWEN_CUA_VERTY_STUCK_RUN", "3") or 3)),
            timeout_s=float(os.getenv("QWEN_CUA_VERTY_TIMEOUT", "5") or 5),
        )


@dataclass(frozen=True, slots=True)
class VertyVerdict:
    verdict: str
    extent: str
    reason: str
    external_change: bool
    no_novel_run: int
    changed_frac: float
    ms: float
    raw: dict[str, Any]

    @property
    def contradicts_action(self) -> bool:
        return self.verdict == "unexpected"


class VertyClient:
    def __init__(self, settings: VertySettings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.settings.timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def reset(self, session: str) -> None:
        if not self.settings.enabled:
            return
        try:
            http = await self._http()
            await http.post(f"{self.settings.url}/session/reset",
                            params={"session": session})
        except Exception:
            # A guard that cannot reach its service simply stops guarding.
            pass

    async def check(
        self,
        *,
        session: str,
        before: str,
        after: str,
        action: str,
        x: int | None = None,
        y: int | None = None,
        dy: int | None = None,
        text: str | None = None,
    ) -> VertyVerdict | None:
        if not self.settings.enabled:
            return None
        params: dict[str, Any] = {
            "session": session, "before": before, "after": after, "action": action,
        }
        if x is not None and y is not None:
            params["x"], params["y"] = int(x), int(y)
        if dy is not None:
            params["dy"] = int(dy)
        if text:
            # Only its presence matters to the verifier, and the full text may be
            # a password.
            params["text"] = text[:64]
        try:
            http = await self._http()
            r = await http.post(f"{self.settings.url}/transition?{urlencode(params)}")
            r.raise_for_status()
            d = r.json()
        except Exception:
            return None
        if "verdict" not in d:
            return None
        return VertyVerdict(
            verdict=str(d.get("verdict", "unknown")),
            extent=str(d.get("extent", "")),
            reason=str(d.get("reason", "")),
            external_change=bool(d.get("external_change", False)),
            no_novel_run=int(d.get("no_novel_run", 0)),
            changed_frac=float(d.get("changed_frac", 0.0)),
            ms=float(d.get("ms", 0.0)),
            raw=d,
        )
