from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, cast

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk

from .config import Settings

ModelProgressCallback = Callable[[str], Awaitable[None]]


class QwenModelClient:
    """OpenAI-compatible multimodal Chat Completions client."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=180,
            max_retries=0,
        )

    async def close(self) -> None:
        await self.client.close()

    async def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        on_progress: ModelProgressCallback | None = None,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.model_retries + 1):
            try:
                return await self._generate_once(
                    model=model,
                    messages=messages,
                    on_progress=on_progress,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self.settings.model_retries:
                    break
                if on_progress is not None:
                    await on_progress(
                        f"Model request failed; retrying ({attempt}/"
                        f"{self.settings.model_retries}): {exc}"
                    )
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        raise last_error

    async def _generate_once(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        on_progress: ModelProgressCallback | None,
    ) -> str:
        extra_body: dict[str, Any] = {"top_k": self.settings.top_k}
        chat_template_kwargs: dict[str, Any] = {}
        if not self.settings.enable_thinking:
            extra_body["enable_thinking"] = False
            chat_template_kwargs["enable_thinking"] = False
        if self.settings.reasoning_effort:
            # Qwen3.8 reads reasoning_effort, not enable_thinking. Leaving this
            # unset lets the server default apply.
            chat_template_kwargs["reasoning_effort"] = self.settings.reasoning_effort
        if chat_template_kwargs:
            extra_body["chat_template_kwargs"] = chat_template_kwargs

        stream = cast(
            AsyncStream[ChatCompletionChunk],
            await self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                stream=True,
                # vLLM emits a final usage-only chunk when asked. Without it a
                # run has no token accounting at all, and token cost is half of
                # what the efficiency experiment is trying to measure.
                stream_options={"include_usage": True},
                extra_body=extra_body,
            ),
        )
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        token_counter = 0
        usage: dict[str, int] = {}
        finish_reason: str | None = None

        async for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = getattr(chunk_usage, field, None)
                    if isinstance(value, int):
                        usage[field] = value
            if not chunk.choices:
                continue
            if getattr(chunk.choices[0], "finish_reason", None):
                finish_reason = str(chunk.choices[0].finish_reason)
            delta = chunk.choices[0].delta
            content = _content_text(getattr(delta, "content", None))
            model_extra = getattr(delta, "model_extra", None) or {}
            reasoning = _content_text(
                getattr(delta, "reasoning_content", None)
                or model_extra.get("reasoning_content")
                or getattr(delta, "reasoning", None)
                or model_extra.get("reasoning")
            )
            if reasoning:
                reasoning_parts.append(reasoning)
            if content:
                content_parts.append(content)
            token_counter += 1
            if on_progress is not None and token_counter % 64 == 0:
                generated_chars = len("".join(reasoning_parts)) + len("".join(content_parts))
                await on_progress(f"Model is generating… {generated_chars:,} characters")

        self.last_usage = dict(usage)
        # `length` means the model was cut off mid-thought. The text still looks
        # like an ordinary assistant message, so without this the run simply ends
        # with "model returned a final assistant message" and the real cause --
        # an exhausted token budget -- is invisible.
        self.last_finish_reason = finish_reason
        reasoning_text = "".join(reasoning_parts).strip()
        content_text = "".join(content_parts).strip()
        if reasoning_text and content_text:
            return f"<think>\n{reasoning_text}\n</think>\n{content_text}"
        return content_text or reasoning_text


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(item.get("text", "")) for item in value if isinstance(item, dict))
    return str(value)
