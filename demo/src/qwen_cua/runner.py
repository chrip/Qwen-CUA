from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .actions import (
    CallUserAction,
    ComputerAction,
    TerminateAction,
    action_to_public_dict,
)
from .computer import BrowserComputer, Screenshot
from .config import Settings
from .model_client import QwenModelClient
from .models import (
    ApprovalRequest,
    BrowserMode,
    EventLevel,
    ModelInfo,
    PendingIntervention,
    RunDetail,
    RunEvent,
    RunOutcome,
    RunStatus,
    RunSummary,
    ScenarioManifest,
    ScreenshotArtifact,
    StartRunRequest,
    StartRunResponse,
    UserInputRequest,
    VerificationStatus,
)
from .protocol import (
    COLLAPSED_SCREENSHOT_TEXT,
    ToolCallParseError,
    build_system_prompt,
    parse_tool_calls,
    redact_tool_text,
    repair_instruction,
)
from .verty import VertyClient, VertySettings, VertyVerdict
from .safety import (
    SafetyIntervention,
    classify_sensitive_action,
    validate_custom_url,
)
from .scenarios import get_scenario, list_scenarios, verify_scenario


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RunRejectedError(RuntimeError):
    pass


@dataclass(slots=True)
class AgentHistory:
    prompt: str
    history_n: int
    image_max: int
    allow_batch: bool = False
    screenshots: list[str] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)
    action_summaries: list[str] = field(default_factory=list)
    feedback: dict[int, str] = field(default_factory=dict)
    # Frames whose transition a cheap visual check confirmed was the expected
    # outcome of the action, with a one-line description of what changed. Such a
    # frame can be folded to that text instead of shipped as an image: the model
    # already knows what it was going to look like, because it asked for it and
    # the pixels agree.
    #
    # The NEWEST frame is never folded, whatever its verdict -- it is the state
    # the model has to act on next.
    foldable: dict[int, str] = field(default_factory=dict)

    def add_screenshot(self, payload: bytes, feedback: str = "",
                       foldable: str = "") -> None:
        self.screenshots.append(_process_image(payload))
        index = len(self.screenshots) - 1
        if feedback:
            self.feedback[index] = feedback
        if foldable:
            self.foldable[index] = foldable

    def add_response(self, response: str, actions: list[ComputerAction]) -> None:
        self.responses.append(response)
        self.action_summaries.append(
            json.dumps(
                [action_to_public_dict(action, redact_text=True) for action in actions],
                ensure_ascii=False,
            )
        )

    def build_messages(self) -> list[dict[str, Any]]:
        total = len(self.screenshots)
        start = max(0, total - self.history_n)
        collapsed_before = max(0, total - self.image_max)
        earlier_actions = self.action_summaries[:start]
        prompt = (
            "Please generate the next move according to the UI screenshot, instruction "
            "and previous actions.\n\n"
            f"Instruction: {self.prompt}\n\n"
            "Previous actions from omitted turns:\n"
            f"{chr(10).join(earlier_actions) if earlier_actions else 'None'}"
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text",
                             "text": build_system_prompt(self.allow_batch)}],
            }
        ]
        for index in range(start, total):
            content: list[dict[str, Any]] = []
            if index == start:
                content.append({"type": "text", "text": prompt})
            elif feedback := self.feedback.get(index):
                content.append(
                    {
                        "type": "text",
                        "text": f"<tool_response>\n{feedback}\n</tool_response>",
                    }
                )
            if index < collapsed_before:
                content.append({"type": "text", "text": COLLAPSED_SCREENSHOT_TEXT})
            elif index < total - 1 and index in self.foldable:
                content.append({"type": "text",
                                "text": f"{COLLAPSED_SCREENSHOT_TEXT} {self.foldable[index]}"})
            else:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{self.screenshots[index]}"},
                    }
                )
            messages.append({"role": "user", "content": content})
            if index < len(self.responses):
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": self.responses[index],
                            }
                        ],
                    }
                )
        return messages


@dataclass(slots=True)
class RunContext:
    detail: RunDetail
    events: list[RunEvent]
    run_dir: Path
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    task: asyncio.Task[None] | None = None
    computer: BrowserComputer | None = None
    pending_future: asyncio.Future[dict[str, Any]] | None = None
    pending_element_ref: str | None = None
    action_count: int = 0
    turn_count: int = 0


class RunnerManager:
    def __init__(
        self,
        settings: Settings,
        *,
        model_client: QwenModelClient | None = None,
        computer_factory: Callable[..., BrowserComputer] | None = None,
    ):
        self.settings = settings
        self.model_client = model_client or QwenModelClient(settings)
        self.computer_factory = computer_factory or BrowserComputer
        self.runs: dict[str, RunContext] = {}
        self._lock = asyncio.Lock()
        self.settings.data_root.mkdir(parents=True, exist_ok=True)
        # Optional cheap visual guard between chained actions. Disabled unless
        # QWEN_CUA_VERTY_URL is set; with it unset the runner behaves exactly as
        # it did before.
        self.verty = VertyClient(VertySettings.from_env())

    async def close(self) -> None:
        tasks = [context.task for context in self.runs.values() if context.task]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self.model_client.close()
        await self.verty.aclose()

    def models(self) -> list[ModelInfo]:
        return [
            ModelInfo(id=model, is_default=model == self.settings.default_model)
            for model in self.settings.models
        ]

    def scenarios(self) -> list[ScenarioManifest]:
        return list_scenarios()

    async def start_run(self, request: StartRunRequest) -> StartRunResponse:
        async with self._lock:
            active = sum(
                context.detail.status
                in {
                    RunStatus.QUEUED,
                    RunStatus.RUNNING,
                    RunStatus.WAITING_FOR_APPROVAL,
                    RunStatus.WAITING_FOR_USER,
                }
                for context in self.runs.values()
            )
            if active >= self.settings.max_concurrent_runs:
                raise ValueError(
                    f"runner already has {active} active run(s); "
                    f"limit={self.settings.max_concurrent_runs}"
                )
            model = request.model or self.settings.default_model
            if model not in self.settings.models:
                raise ValueError(f"model is not in QWEN_CUA_MODELS: {model}")
            if request.browser_mode is BrowserMode.HEADFUL and self.settings.docker_mode:
                raise ValueError("headful browser mode is unavailable in Docker")

            scenario = get_scenario(request.scenario_id) if request.scenario_id else None
            if request.scenario_id and scenario is None:
                raise ValueError(f"unknown scenario: {request.scenario_id}")
            if scenario is not None:
                target_url = f"http://127.0.0.1:{self.settings.port}/labs/{scenario.lab_path}"
            else:
                target_url = str(request.target_url)
                validate_custom_url(
                    target_url,
                    allow_private=self.settings.allow_private_urls,
                )

            run_id = uuid.uuid4().hex
            run_dir = self.settings.data_root / "runs" / run_id
            for child in ("screenshots", "downloads", "uploads"):
                (run_dir / child).mkdir(parents=True, exist_ok=True)
            detail = RunDetail(
                id=run_id,
                status=RunStatus.QUEUED,
                prompt=request.prompt,
                model=model,
                scenario_id=request.scenario_id,
                target_url=target_url,
                browser_mode=request.browser_mode,
                max_turns=request.max_turns or self.settings.default_max_turns,
                started_at=_now(),
                event_stream_url=f"/api/runs/{run_id}/events",
                replay_url=f"/api/runs/{run_id}/replay",
            )
            context = RunContext(detail=detail, events=[], run_dir=run_dir)
            self.runs[run_id] = context
            await self._persist(context)
            context.task = asyncio.create_task(self._execute(context))
            return StartRunResponse(
                run_id=run_id,
                status=detail.status,
                event_stream_url=detail.event_stream_url,
                replay_url=detail.replay_url,
            )

    async def get_run(self, run_id: str) -> RunDetail:
        return self._context(run_id).detail.model_copy(deep=True)

    async def get_events(self, run_id: str) -> list[RunEvent]:
        return [event.model_copy(deep=True) for event in self._context(run_id).events]

    async def stop_run(self, run_id: str) -> RunDetail:
        context = self._context(run_id)
        if context.detail.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            return context.detail.model_copy(deep=True)
        if context.task is not None:
            context.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await context.task
        return context.detail.model_copy(deep=True)

    async def resolve_approval(
        self,
        run_id: str,
        intervention_id: str,
        request: ApprovalRequest,
    ) -> RunDetail:
        context = self._context(run_id)
        pending = context.detail.pending_intervention
        if pending is None or pending.id != intervention_id:
            raise ValueError("approval is no longer pending")
        if pending.requires_file and request.decision == "approve":
            raise ValueError("this approval requires a file upload")
        self._resolve_pending(
            context,
            {"decision": request.decision},
        )
        return context.detail.model_copy(deep=True)

    async def resolve_user_input(
        self,
        run_id: str,
        request: UserInputRequest,
    ) -> RunDetail:
        context = self._context(run_id)
        pending = context.detail.pending_intervention
        if pending is None or pending.id != request.intervention_id or pending.kind != "user_input":
            raise ValueError("user input is no longer pending")
        self._resolve_pending(context, {"decision": "approve", "text": request.text})
        return context.detail.model_copy(deep=True)

    async def resolve_file_upload(
        self,
        run_id: str,
        intervention_id: str,
        *,
        filename: str,
        payload: bytes,
    ) -> RunDetail:
        context = self._context(run_id)
        pending = context.detail.pending_intervention
        if pending is None or pending.id != intervention_id or pending.kind != "file_upload":
            raise ValueError("file upload is no longer pending")
        if len(payload) > self.settings.max_upload_bytes:
            raise ValueError(f"file exceeds {self.settings.max_upload_bytes} byte upload limit")
        safe_name = Path(filename).name or "upload.bin"
        destination = context.run_dir / "uploads" / f"{uuid.uuid4().hex}-{safe_name}"
        await asyncio.to_thread(destination.write_bytes, payload)
        self._resolve_pending(
            context,
            {"decision": "approve", "file_path": str(destination)},
        )
        return context.detail.model_copy(deep=True)

    async def wait_for_events(
        self,
        run_id: str,
        *,
        after_sequence: int,
    ) -> AsyncIterator[RunEvent | None]:
        context = self._context(run_id)
        cursor = max(0, after_sequence + 1)
        while True:
            while cursor < len(context.events):
                event = context.events[cursor]
                cursor += 1
                yield event
            if context.detail.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                break
            async with context.condition:
                try:
                    await asyncio.wait_for(context.condition.wait(), timeout=15)
                except TimeoutError:
                    yield None

    async def replay(self, run_id: str) -> dict[str, Any]:
        context = self._context(run_id)
        return self._replay_payload(context)

    def screenshot_path(self, run_id: str, filename: str) -> Path:
        context = self._context(run_id)
        path = context.run_dir / "screenshots" / Path(filename).name
        if not path.is_file():
            raise FileNotFoundError(filename)
        return path

    def download_path(self, run_id: str, filename: str) -> Path:
        context = self._context(run_id)
        path = context.run_dir / "downloads" / Path(filename).name
        if not path.is_file():
            raise FileNotFoundError(filename)
        return path

    async def _execute(self, context: RunContext) -> None:
        history = AgentHistory(
            prompt=context.detail.prompt,
            history_n=self.settings.history_n,
            image_max=self.settings.image_max,
            allow_batch=self.settings.allow_batch,
        )
        trusted_scenario = context.detail.scenario_id is not None
        computer = self.computer_factory(
            browser_mode=context.detail.browser_mode,
            width=self.settings.viewport_width,
            height=self.settings.viewport_height,
            downloads_dir=context.run_dir / "downloads",
        )
        context.computer = computer
        try:
            context.detail.status = RunStatus.RUNNING
            await self._emit(
                context,
                type_="run_started",
                level=EventLevel.OK,
                message="Run started.",
                detail={
                    "model": context.detail.model,
                    "target_url": context.detail.target_url,
                    "browser_mode": context.detail.browser_mode,
                },
            )
            await computer.start()
            await computer.navigate(context.detail.target_url)
            initial = await self._capture(context, "initial")
            history.add_screenshot(initial.payload)

            for turn in range(1, context.detail.max_turns + 1):
                context.turn_count = turn
                await self._emit(
                    context,
                    type_="model_turn_started",
                    level=EventLevel.PENDING,
                    message=f"Model turn {turn} started.",
                )
                messages = history.build_messages()
                response = await self.model_client.generate(
                    model=context.detail.model,
                    messages=messages,
                    on_progress=lambda text: self._emit(
                        context,
                        type_="model_progress",
                        level=EventLevel.PENDING,
                        message=text,
                    ),
                )
                try:
                    actions = parse_tool_calls(response)
                except ToolCallParseError as exc:
                    repair_messages = [
                        *messages,
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": repair_instruction(str(exc))},
                    ]
                    await self._emit(
                        context,
                        type_="tool_call_repair",
                        level=EventLevel.WARN,
                        message="Malformed tool call; requesting one repair.",
                        detail=str(exc),
                    )
                    response = await self.model_client.generate(
                        model=context.detail.model,
                        messages=repair_messages,
                    )
                    actions = parse_tool_calls(response)

                await self._emit(
                    context,
                    type_="model_response",
                    level=EventLevel.OK,
                    message="Model response received.",
                    detail=redact_tool_text(response),
                )
                # Token accounting per model turn. Half of what the efficiency
                # experiment measures is cost, and cost is tokens, not turns.
                usage = getattr(self.model_client, "last_usage", None)
                if usage:
                    await self._emit(
                        context,
                        type_="model_usage",
                        level=EventLevel.OK,
                        message=(f"Tokens: {usage.get('prompt_tokens', 0)} prompt / "
                                 f"{usage.get('completion_tokens', 0)} completion"),
                        detail=dict(usage),
                    )
                history.add_response(response, actions)
                if not actions:
                    # Distinguish "the model decided to stop" from "the model was
                    # cut off before it could act". Both arrive here as a message
                    # with no tool call, and only one of them is a real ending.
                    truncated = getattr(self.model_client, "last_finish_reason", None) == "length"

                    # A cut-off turn is the model failing to answer, not choosing
                    # to stop. Ending the run there throws away a task that was
                    # going fine -- measured: it killed 5 of 8 experiment runs
                    # across two configurations. Tell it what went wrong, capture
                    # the current screen, and give it the next turn.
                    if truncated:
                        await self._emit(
                            context,
                            type_="model_truncated",
                            level=EventLevel.ERROR,
                            message=(f"Model hit the {self.settings.max_tokens}-token "
                                     f"budget without emitting a tool call; retrying."),
                            detail={"max_tokens": self.settings.max_tokens},
                        )
                        recovery = await self._capture(context, f"turn-{turn}-truncated")
                        history.add_screenshot(
                            recovery.payload,
                            feedback=(
                                "Your previous response was cut off before it contained a "
                                "tool call, so nothing was executed. Reply with exactly one "
                                "<tool_call> block and no other text."
                            ),
                        )
                        continue

                    note = (
                        "Model returned a final assistant message."
                    )
                    await self._complete_from_current_state(
                        context,
                        requested_outcome=None,
                        note=note,
                    )
                    return

                terminal: TerminateAction | None = None
                last_screenshot: Screenshot | None = None
                feedback_parts: list[str] = []
                turn_fold: str = ""   # set when the turn ended in a verified state
                for action in actions:
                    if isinstance(action, CallUserAction):
                        answer = await self._await_user_input(context, action)
                        feedback_parts.append(f"User response to call_user: {answer}")
                        continue
                    if isinstance(action, TerminateAction):
                        terminal = action
                        feedback_parts.append(f"Model requested terminate({action.status}).")
                        break

                    inspection = await computer.inspect_target(action)
                    safety = (
                        None
                        if trusted_scenario
                        else classify_sensitive_action(
                            action=action,
                            inspection=inspection,
                            current_url=computer.current_url(),
                        )
                    )
                    public_action = action_to_public_dict(
                        action,
                        redact_text=bool(inspection and inspection.is_password),
                    )
                    await self._emit(
                        context,
                        type_="action_requested",
                        level=EventLevel.PENDING,
                        message=f"Action requested: {action.action}",
                        detail=public_action,
                    )
                    # A model can emit an action the browser refuses -- a key spec
                    # like "Down Down Enter", a coordinate outside the viewport.
                    # That is the model being wrong, not the run being over: an
                    # agent that cannot survive its own malformed actions will
                    # never finish a long task. Report it back and let the model
                    # decide again, the same way a guard stop does.
                    try:
                        if safety is not None:
                            resolution = await self._await_approval(context, safety)
                            if resolution["decision"] != "approve":
                                raise RunRejectedError("Operator rejected a sensitive action.")
                            if safety.kind == "file_upload":
                                file_path = Path(str(resolution["file_path"]))
                                await computer.set_input_file(safety.element_ref, file_path)
                            else:
                                await computer.execute(action)
                        else:
                            await computer.execute(action)
                    except (RunRejectedError, asyncio.CancelledError):
                        raise
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}".split("\n")[0][:300]
                        await self._emit(
                            context,
                            type_="action_rejected",
                            level=EventLevel.ERROR,
                            message=f"Action rejected by the browser: {message}",
                            detail={"action": public_action, "error": message},
                        )
                        feedback_parts.append(
                            f"The {action.action} action was rejected: {message}. "
                            f"It was not performed. Choose a different action."
                        )
                        break
                    context.action_count += 1
                    before_path = self._screenshot_path(context, -1)
                    last_screenshot = await self._capture(
                        context,
                        f"turn-{turn}-action-{context.action_count}",
                    )
                    feedback_parts.append(f"Executed {action.action}.")
                    await self._emit(
                        context,
                        type_="action_completed",
                        level=EventLevel.OK,
                        message=f"Action completed: {action.action}",
                        detail=public_action,
                        screenshot_id=context.detail.screenshots[-1].id,
                    )

                    # ── Verty guard ──────────────────────────────────────────
                    # The model planned this whole chain against the screen it
                    # saw at the start of the turn. Check the assumption held
                    # before running the next link.
                    guard, turn_fold = await self._verty_check(
                        context, action, public_action, before_path
                    )
                    if guard is not None:
                        feedback_parts.append(guard)
                        break

                if terminal is not None:
                    await self._complete_from_current_state(
                        context,
                        requested_outcome=terminal.status,
                        note=f"Model requested terminate({terminal.status}).",
                    )
                    return
                if last_screenshot is None:
                    last_screenshot = await self._capture(context, f"turn-{turn}-feedback")
                history.add_screenshot(
                    last_screenshot.payload,
                    feedback="\n".join(feedback_parts),
                    foldable=turn_fold if self.verty.settings.fold else "",
                )

            await self._fail(
                context,
                f"Run exhausted the configured {context.detail.max_turns}-turn budget.",
            )
        except asyncio.CancelledError:
            context.detail.status = RunStatus.CANCELLED
            context.detail.completed_at = _now()
            context.detail.summary = RunSummary(
                outcome=RunOutcome.FAILURE,
                verification=VerificationStatus.NOT_RUN,
                turns=context.turn_count,
                actions=context.action_count,
                screenshots=len(context.detail.screenshots),
                notes=["Run cancelled by operator."],
            )
            await self._emit(
                context,
                type_="run_cancelled",
                level=EventLevel.WARN,
                message="Run cancelled.",
            )
            raise
        except RunRejectedError as exc:
            context.detail.status = RunStatus.CANCELLED
            context.detail.completed_at = _now()
            context.detail.summary = RunSummary(
                outcome=RunOutcome.FAILURE,
                verification=VerificationStatus.NOT_RUN,
                turns=context.turn_count,
                actions=context.action_count,
                screenshots=len(context.detail.screenshots),
                notes=[str(exc)],
            )
            await self._emit(
                context,
                type_="run_cancelled",
                level=EventLevel.WARN,
                message=str(exc),
            )
        except Exception as exc:
            await self._fail(context, str(exc))
        finally:
            await computer.close()
            context.computer = None
            await self._persist(context)

    async def _complete_from_current_state(
        self,
        context: RunContext,
        *,
        requested_outcome: str | None,
        note: str,
    ) -> None:
        verification = VerificationStatus.NOT_APPLICABLE
        verification_detail: str | None = None
        if requested_outcome == "failure":
            outcome = RunOutcome.FAILURE
            verification = VerificationStatus.NOT_RUN
        elif context.detail.scenario_id and context.computer:
            result = verify_scenario(
                context.detail.scenario_id,
                await context.computer.read_lab_state(),
            )
            verification = VerificationStatus.PASSED if result.passed else VerificationStatus.FAILED
            verification_detail = result.detail
            outcome = RunOutcome.SUCCESS if result.passed else RunOutcome.FAILURE
            await self._emit(
                context,
                type_="verification_completed",
                level=EventLevel.OK if result.passed else EventLevel.ERROR,
                message=(
                    "Scenario verification passed."
                    if result.passed
                    else "Scenario verification failed."
                ),
                detail=result.detail,
            )
        else:
            outcome = RunOutcome.UNVERIFIED

        context.detail.status = RunStatus.COMPLETED
        context.detail.completed_at = _now()
        context.detail.summary = RunSummary(
            outcome=outcome,
            verification=verification,
            verification_detail=verification_detail,
            turns=context.turn_count,
            actions=context.action_count,
            screenshots=len(context.detail.screenshots),
            notes=[note],
        )
        await self._emit(
            context,
            type_="run_completed",
            level=EventLevel.OK if outcome is not RunOutcome.FAILURE else EventLevel.ERROR,
            message=f"Run completed with outcome: {outcome.value}.",
            detail=context.detail.summary.model_dump(mode="json"),
        )

    async def _fail(self, context: RunContext, message: str) -> None:
        # Capture whatever the lab knows before tearing the run down. A run that
        # exhausts its turn budget never reaches verification, so any state the
        # lab was recording -- for an experiment, that is often the ground truth
        # the run existed to produce -- would otherwise be lost with the browser.
        try:
            if context.computer is not None:
                lab_state = await context.computer.read_lab_state()
                if lab_state:
                    await self._emit(
                        context,
                        type_="lab_state",
                        level=EventLevel.OK,
                        message="Lab state captured before failure.",
                        detail=lab_state,
                    )
        except Exception:
            pass
        context.detail.status = RunStatus.FAILED
        context.detail.completed_at = _now()
        context.detail.summary = RunSummary(
            outcome=RunOutcome.FAILURE,
            verification=VerificationStatus.NOT_RUN,
            turns=context.turn_count,
            actions=context.action_count,
            screenshots=len(context.detail.screenshots),
            notes=[message],
        )
        await self._emit(
            context,
            type_="run_failed",
            level=EventLevel.ERROR,
            message="Run failed.",
            detail=message,
        )

    async def _verty_check(
        self,
        context: RunContext,
        action: ComputerAction,
        public_action: dict[str, Any],
        before_path: str | None,
    ) -> tuple[str | None, str]:
        """Cheap visual check after one executed action.

        Returns (stop_feedback, fold_text):
          stop_feedback  text to give the model, and stop the chain; None to continue
          fold_text      a one-line description of a VERIFIED transition, which
                         lets the resulting frame be folded to text in later
                         turns instead of shipped as an image
        """
        if not self.verty.settings.enabled or before_path is None:
            return None, ""
        after_path = self._screenshot_path(context, -1)
        if after_path is None or after_path == before_path:
            return None, ""

        coord = public_action.get("coordinate")
        x = y = None
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            shot = context.detail.screenshots[-1]
            # The protocol's 0..999 grid -> frame pixels, which is what the
            # verifier measures in.
            x = int(round(float(coord[0]) / 1000.0 * shot.width))
            y = int(round(float(coord[1]) / 1000.0 * shot.height))
        pixels = public_action.get("pixels")
        # computer.py executes scroll as wheel(0, -pixels), so a positive
        # `pixels` scrolls the view UP, which is a negative dy for the verifier.
        dy = -int(pixels) if isinstance(pixels, (int, float)) and action.action == "scroll" else None

        verdict = await self.verty.check(
            session=context.detail.id,
            before=before_path,
            after=after_path,
            action=str(public_action.get("action", "")),
            x=x, y=y, dy=dy,
            text=public_action.get("text") if isinstance(public_action.get("text"), str) else None,
        )
        if verdict is None:
            return None, ""

        detail = {
            "verdict": verdict.verdict,
            "extent": verdict.extent,
            "reason": verdict.reason,
            "external_change": verdict.external_change,
            "no_novel_run": verdict.no_novel_run,
            "ms": verdict.ms,
        }
        await self._emit(
            context,
            type_="verty_check",
            level=EventLevel.OK if verdict.verdict != "unexpected" else EventLevel.PENDING,
            message=f"Visual check: {verdict.verdict} ({verdict.reason})",
            detail=detail,
        )

        # A transition the check confirms was the action's expected outcome, with
        # nothing unexplained alongside it, needs no image in later turns.
        fold = ""
        if verdict.verdict == "expected" and not verdict.external_change:
            fold = (f"Verified: {public_action.get('action')} produced its expected "
                    f"result ({verdict.reason}).")

        if not self.verty.settings.guard:
            return None, fold

        # (1) The action demonstrably did not do what it should have. Every
        # later link in the chain was planned assuming it did.
        if verdict.contradicts_action:
            await self._emit(
                context,
                type_="verty_guard_stop",
                level=EventLevel.PENDING,
                message=f"Chain stopped: {verdict.reason}",
                detail=detail,
            )
            return (
                f"A visual check found that {public_action.get('action')} did not take "
                f"effect: {verdict.reason}. Remaining planned actions were skipped "
                f"because they assumed it had. Look at the current screenshot and "
                f"decide again."
            ), ""

        # (2) Nothing new has appeared for several steps. Change is still
        # happening -- focus rings, carets -- but the screen keeps showing states
        # it has shown before, which is what going in circles looks like.
        run = self.verty.settings.stuck_run
        if run and verdict.no_novel_run >= run:
            await self._emit(
                context,
                type_="verty_guard_stop",
                level=EventLevel.PENDING,
                message=f"Chain stopped: {verdict.no_novel_run} steps with no new content",
                detail=detail,
            )
            return (
                f"A visual check found that the last {verdict.no_novel_run} actions "
                f"produced no content that had not already been on screen. The "
                f"current approach is not making progress; try a different one."
            ), ""

        return None, fold

    def _screenshot_path(self, context: RunContext, index: int = -1) -> str | None:
        """On-disk path of a captured screenshot. verty-serve reads paths, not bodies."""
        shots = context.detail.screenshots
        if not shots or abs(index) > len(shots):
            return None
        return str(context.run_dir / "screenshots" / f"{shots[index].sequence:04d}.png")

    async def _capture(
        self,
        context: RunContext,
        label: str,
    ) -> Screenshot:
        if context.computer is None:
            raise RuntimeError("browser is not available")
        screenshot = await context.computer.capture()
        sequence = len(context.detail.screenshots) + 1
        filename = f"{sequence:04d}.png"
        await asyncio.to_thread(
            (context.run_dir / "screenshots" / filename).write_bytes,
            screenshot.payload,
        )
        artifact = ScreenshotArtifact(
            id=f"screenshot-{sequence}",
            sequence=sequence,
            label=label,
            captured_at=_now(),
            page_url=screenshot.page_url,
            page_title=screenshot.page_title,
            width=screenshot.width,
            height=screenshot.height,
            url=f"/api/runs/{context.detail.id}/artifacts/screenshots/{filename}",
        )
        context.detail.screenshots.append(artifact)
        await self._emit(
            context,
            type_="screenshot_captured",
            level=EventLevel.OK,
            message=f"Screenshot captured: {label}",
            detail=artifact.model_dump(mode="json"),
            screenshot_id=artifact.id,
        )
        return screenshot

    async def _await_approval(
        self,
        context: RunContext,
        safety: SafetyIntervention,
    ) -> dict[str, Any]:
        intervention = PendingIntervention(
            id=uuid.uuid4().hex,
            kind=safety.kind,  # type: ignore[arg-type]
            title=safety.title,
            message=safety.message,
            created_at=_now(),
            action=safety.action,
            requires_file=safety.requires_file,
        )
        context.detail.pending_intervention = intervention
        context.pending_element_ref = safety.element_ref
        context.pending_future = asyncio.get_running_loop().create_future()
        context.detail.status = RunStatus.WAITING_FOR_APPROVAL
        await self._emit(
            context,
            type_="approval_required",
            level=EventLevel.WARN,
            message=safety.title,
            detail=intervention.model_dump(mode="json"),
        )
        resolution = await context.pending_future
        context.pending_future = None
        context.pending_element_ref = None
        context.detail.pending_intervention = None
        context.detail.status = RunStatus.RUNNING
        await self._emit(
            context,
            type_="approval_resolved",
            level=(EventLevel.OK if resolution.get("decision") == "approve" else EventLevel.WARN),
            message=f"Approval {resolution.get('decision')}.",
        )
        return resolution

    async def _await_user_input(
        self,
        context: RunContext,
        action: CallUserAction,
    ) -> str:
        intervention = PendingIntervention(
            id=uuid.uuid4().hex,
            kind="user_input",
            title="The agent has a question",
            message=action.text,
            created_at=_now(),
            action=action_to_public_dict(action),
        )
        context.detail.pending_intervention = intervention
        context.pending_future = asyncio.get_running_loop().create_future()
        context.detail.status = RunStatus.WAITING_FOR_USER
        await self._emit(
            context,
            type_="user_input_required",
            level=EventLevel.WARN,
            message=action.text,
            detail=intervention.model_dump(mode="json"),
        )
        resolution = await context.pending_future
        context.pending_future = None
        context.detail.pending_intervention = None
        context.detail.status = RunStatus.RUNNING
        return str(resolution["text"])

    def _resolve_pending(
        self,
        context: RunContext,
        resolution: dict[str, Any],
    ) -> None:
        future = context.pending_future
        if future is None or future.done():
            raise ValueError("intervention is no longer pending")
        future.set_result(resolution)

    async def _emit(
        self,
        context: RunContext,
        *,
        type_: str,
        level: EventLevel,
        message: str,
        detail: Any | None = None,
        screenshot_id: str | None = None,
    ) -> None:
        event = RunEvent(
            id=f"{context.detail.id}:{len(context.events)}",
            run_id=context.detail.id,
            sequence=len(context.events),
            type=type_,
            level=level,
            message=message,
            detail=detail,
            screenshot_id=screenshot_id,
            created_at=_now(),
        )
        context.events.append(event)
        await self._persist(context)
        async with context.condition:
            context.condition.notify_all()

    async def _persist(self, context: RunContext) -> None:
        detail_json = context.detail.model_dump_json(indent=2)
        events_text = "".join(f"{event.model_dump_json()}\n" for event in context.events)
        replay_text = json.dumps(
            self._replay_payload(context),
            ensure_ascii=False,
            indent=2,
        )
        await asyncio.gather(
            asyncio.to_thread(
                (context.run_dir / "run.json").write_text,
                detail_json,
                encoding="utf-8",
            ),
            asyncio.to_thread(
                (context.run_dir / "events.jsonl").write_text,
                events_text,
                encoding="utf-8",
            ),
            asyncio.to_thread(
                (context.run_dir / "replay.json").write_text,
                replay_text,
                encoding="utf-8",
            ),
        )

    def _replay_payload(self, context: RunContext) -> dict[str, Any]:
        return {
            "version": 1,
            "run": context.detail.model_dump(mode="json"),
            "events": [event.model_dump(mode="json") for event in context.events],
            "artifacts": {
                "screenshots": [
                    screenshot.model_dump(mode="json") for screenshot in context.detail.screenshots
                ],
                "downloads": (
                    [path.name for path in (context.run_dir / "downloads").iterdir()]
                    if (context.run_dir / "downloads").exists()
                    else []
                ),
            },
        }

    def _context(self, run_id: str) -> RunContext:
        try:
            return self.runs[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown run: {run_id}") from exc


def _process_image(payload: bytes) -> str:
    image = Image.open(BytesIO(payload)).convert("RGB")
    width, height = image.size
    max_pixels = 16 * 16 * 4 * 12_800
    if width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        width = max(32, int(width * scale))
        height = max(32, int(height * scale))
    width = max(32, round(width / 32) * 32)
    height = max(32, round(height / 32) * 32)
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
