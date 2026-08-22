from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import ScenarioManifest


@dataclass(frozen=True, slots=True)
class VerificationResult:
    passed: bool
    detail: str


SCENARIOS = (
    ScenarioManifest(
        id="kanban-reprioritize",
        title="Launch Board",
        description="Drag and reorder cards to match a requested sprint board.",
        default_prompt=(
            "Reorganize the Launch Board so Backlog contains “Write release notes”, "
            "In Progress contains “Polish demo” followed by “Run browser tests”, and "
            "Done contains “Tag release”. Verify the final order, then terminate successfully."
        ),
        lab_path="kanban/index.html",
        tags=["drag-drop", "productivity", "verification"],
    ),
    ScenarioManifest(
        id="shifting-checklist",
        title="Release Checklist",
        description=(
            "Tick three checkboxes; each tick pushes the remaining ones further "
            "down the page."
        ),
        default_prompt=(
            "Tick every item in the Release Checklist, then terminate successfully."
        ),
        lab_path="shifting/index.html",
        tags=["clicking", "moving-targets", "verification"],
    ),
    ScenarioManifest(
        id="interrupted-checklist",
        title="Deploy Checklist",
        description=(
            "Tick four checkboxes while an activity feed updates on its own timer."
        ),
        default_prompt=(
            "Tick every item in the Deploy Checklist, then terminate successfully. "
            "Ignore the Activity feed on the right; it updates by itself."
        ),
        lab_path="interrupted/index.html",
        tags=["clicking", "external-change", "verification"],
    ),
    ScenarioManifest(
        id="quiet-checklist",
        title="Deploy Checklist (quiet)",
        description=(
            "The Deploy Checklist with its activity feed frozen -- the control "
            "for whether self-changing content degrades the agent itself."
        ),
        default_prompt=(
            "Tick every item in the Deploy Checklist, then terminate successfully."
        ),
        lab_path="interrupted/index.html?quiet=1",
        tags=["clicking", "control", "verification"],
    ),
    ScenarioManifest(
        id="registration-complete",
        title="Summit Registration",
        description="Complete and submit a multi-step event registration form.",
        default_prompt=(
            "Register Jordan Lee for the Applied Agents Summit using jordan@example.com. "
            "Choose the Developer ticket, select the Browser agents workshop, request a "
            "vegetarian meal, accept the code of conduct, submit the form, verify the "
            "confirmation, then terminate successfully."
        ),
        lab_path="registration/index.html",
        tags=["forms", "typing", "verification"],
    ),
)


def list_scenarios() -> list[ScenarioManifest]:
    return [scenario.model_copy(deep=True) for scenario in SCENARIOS]


def get_scenario(scenario_id: str) -> ScenarioManifest | None:
    return next(
        (scenario.model_copy(deep=True) for scenario in SCENARIOS if scenario.id == scenario_id),
        None,
    )


def verify_scenario(scenario_id: str, state: dict[str, Any] | None) -> VerificationResult:
    if state is None:
        return VerificationResult(False, "The lab did not expose a verification state.")
    if scenario_id in ("interrupted-checklist", "quiet-checklist"):
        checked = state.get("checked") or {}
        if not checked:
            return VerificationResult(False, "The lab exposed no checklist state.")
        missing = [k for k, ok in checked.items() if not ok]
        # Carry the injection timestamps into the run record. They are the
        # ground truth an external-change detector is scored against, and the
        # verification detail is the only channel that survives into run.json.
        stamps = [int(x.get("at", 0)) for x in (state.get("injections") or [])]
        detail = f"injections={len(stamps)} at={stamps}"
        if missing:
            return VerificationResult(False, f"unchecked={sorted(missing)}; {detail}")
        return VerificationResult(True, f"all checked; {detail}")
    if scenario_id == "shifting-checklist":
        checked = state.get("checked") or {}
        missing = [k for k, v in checked.items() if not v]
        if not checked:
            return VerificationResult(False, "The lab exposed no checklist state.")
        if missing:
            return VerificationResult(False, f"unchecked={sorted(missing)}")
        return VerificationResult(
            True, f"all checked; observed shift={state.get('shifted')}"
        )
    if scenario_id == "kanban-reprioritize":
        expected_columns: dict[str, Any] = {
            "backlog": ["release-notes"],
            "progress": ["polish-demo", "browser-tests"],
            "done": ["tag-release"],
        }
        observed = state.get("columns")
        return VerificationResult(
            observed == expected_columns,
            f"expected={expected_columns}; observed={observed}",
        )
    if scenario_id == "registration-complete":
        expected_registration: dict[str, Any] = {
            "name": "Jordan Lee",
            "email": "jordan@example.com",
            "ticket": "developer",
            "workshop": "browser-agents",
            "meal": "vegetarian",
            "conduct": True,
            "submitted": True,
        }
        observed = {key: state.get(key) for key in expected_registration}
        return VerificationResult(
            observed == expected_registration,
            f"expected={expected_registration}; observed={observed}",
        )
    return VerificationResult(False, f"Unknown scenario: {scenario_id}")
