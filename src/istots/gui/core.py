from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from istots.app.convert import ConvertRequest
from istots.app.setup import SetupRequest
from istots.derived_assets import resolve_derived_mmproj_output_path
from istots.gui.bootstrap_windows import (
    EXTERNAL_RUNTIME_SOURCE,
    load_managed_runtime_state,
    MANAGED_RUNTIME_SOURCE,
    OVERRIDE_RUNTIME_SOURCE,
    describe_runtime_binding,
    gui_managed_models_dir,
    resolve_gui_runtime_binding,
    validate_llama_server_binary,
)
from istots.llama_runtime import (
    DEFAULT_LLAMA_SERVER_HOST,
    LlamaServerOverrides,
    detect_llama_server_path,
    resolve_llama_server_role_assets,
    run_llama_server_doctor,
)
from istots.model_store import (
    DEFAULT_GGUF_FILENAME,
    DEFAULT_GGUF_MMPROJ_FILENAME,
    DEFAULT_GGUF_MODEL_ID,
    resolve_local_model_path,
)
from istots.ocr import LOCAL_PADDLE_CTX_SIZE


@dataclass(frozen=True)
class GuiRuntimeStatus:
    ready: bool
    headline: str
    detail: str
    missing_items: tuple[str, ...]
    runtime_binary_path: Path | None = None
    models_dir: Path | None = None
    runtime_source: str = "missing"
    runtime_release_tag: str | None = None
    runtime_variant_id: str | None = None


@dataclass(frozen=True)
class GuiScreenState:
    runtime_status: GuiRuntimeStatus
    input_sup: Path | None = None
    output_srt: Path | None = None
    enable_furigana_mask: bool = False


@dataclass(frozen=True)
class GuiPrimaryAction:
    kind: str
    label: str
    enabled: bool


def _dedupe_strings(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _default_gui_doctor_overrides() -> LlamaServerOverrides:
    # Keep GUI Test aligned with the retained Paddle llama-server memory policy
    # used by real convert and structured runtime doctor paths.
    return LlamaServerOverrides(ctx_size=LOCAL_PADDLE_CTX_SIZE)


def suggest_output_srt_path(input_sup: Path) -> Path:
    normalized = input_sup.expanduser().resolve()
    candidate = normalized.with_suffix(".srt")
    if not candidate.exists():
        return candidate

    parent = candidate.parent
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 2
    while True:
        numbered = parent / f"{stem} ({counter}){suffix}"
        if not numbered.exists():
            return numbered
        counter += 1


def probe_runtime_status(
    *,
    models_dir: Path | None = None,
    runtime_binary_path: Path | None = None,
    min_pixels: int = 32768,
) -> GuiRuntimeStatus:
    missing_items: list[str] = []
    binding = resolve_gui_runtime_binding(explicit_binary_path=runtime_binary_path)
    effective_models_dir = (
        models_dir.expanduser().resolve()
        if models_dir is not None
        else binding.models_dir.expanduser().resolve()
    )

    binary_path = binding.binary_path if runtime_binary_path is None else detect_llama_server_path(runtime_binary_path)
    if binary_path is None or not binary_path.exists():
        missing_items.append("llama-server")
    runtime_issue_messages = _probe_runtime_validation_messages(
        binding_source=binding.source,
        binary_path=binary_path,
    )
    persisted_issue_messages = _load_persisted_validation_messages(
        binding_source=binding.source,
        binary_path=binary_path,
    )
    runtime_issue_messages = tuple(dict.fromkeys((*persisted_issue_messages, *runtime_issue_messages)))
    if runtime_issue_messages:
        missing_items.append("runtime validation")

    ocr_assets = resolve_llama_server_role_assets("ocr", models_dir=effective_models_dir, min_pixels=min_pixels)
    fast_assets = resolve_llama_server_role_assets("ocr-fast", models_dir=effective_models_dir, min_pixels=min_pixels)

    if not ocr_assets.model_path.exists():
        missing_items.append(DEFAULT_GGUF_FILENAME)
    if not ocr_assets.mmproj_path.exists():
        missing_items.append(DEFAULT_GGUF_MMPROJ_FILENAME)
    if not fast_assets.mmproj_path.exists():
        missing_items.append(fast_assets.mmproj_path.name)

    if missing_items:
        return GuiRuntimeStatus(
            ready=False,
            headline="Setup",
            detail=_format_runtime_status_detail(
                runtime_detail=describe_runtime_binding(binding),
                runtime_binary_path=binding.binary_path,
                models_dir=effective_models_dir,
                missing_items=_dedupe_strings(missing_items),
                issue_messages=runtime_issue_messages,
            ),
            missing_items=_dedupe_strings(missing_items),
            runtime_binary_path=binary_path,
            models_dir=effective_models_dir,
            runtime_source=binding.source,
            runtime_release_tag=binding.release_tag,
            runtime_variant_id=binding.variant_id,
        )

    return GuiRuntimeStatus(
        ready=True,
        headline="Ready",
        detail=_format_runtime_status_detail(
            runtime_detail=describe_runtime_binding(binding),
            runtime_binary_path=binding.binary_path,
            models_dir=effective_models_dir,
            missing_items=(),
            issue_messages=(),
        ),
        missing_items=(),
        runtime_binary_path=binary_path,
        models_dir=effective_models_dir,
        runtime_source=binding.source,
        runtime_release_tag=binding.release_tag,
        runtime_variant_id=binding.variant_id,
    )


def run_gui_doctor_check(
    *,
    models_dir: Path | None = None,
    runtime_binary_path: Path | None = None,
    min_pixels: int = 32768,
    host: str = DEFAULT_LLAMA_SERVER_HOST,
    cancel_event: threading.Event | None = None,
) -> GuiRuntimeStatus:
    issue_labels: list[str] = []
    issue_messages: list[str] = []
    binding = resolve_gui_runtime_binding(explicit_binary_path=runtime_binary_path)
    effective_models_dir = (
        models_dir.expanduser().resolve()
        if models_dir is not None
        else binding.models_dir.expanduser().resolve()
    )
    effective_binary_path = (
        runtime_binary_path.expanduser().resolve()
        if runtime_binary_path is not None
        else binding.binary_path
    )

    for role in ("ocr", "ocr-fast"):
        _raise_if_gui_check_cancelled(cancel_event, stage=f"{role} runtime check")
        report = run_llama_server_doctor(
            role=role,
            models_dir=effective_models_dir,
            min_pixels=min_pixels,
            explicit_binary_path=effective_binary_path,
            host=host,
            overrides=_default_gui_doctor_overrides(),
            cancel_event=cancel_event,
        )
        if report.ok:
            continue

        role_name = str(getattr(report.role, "value", report.role))
        for issue in report.issues:
            issue_messages.append(f"{role_name}: {issue.message}")
            if issue.code == "missing_binary":
                issue_labels.append("llama-server")
            elif issue.code == "missing_model" and report.launch_spec is not None:
                issue_labels.append(report.launch_spec.model_path.name)
            elif issue.code == "missing_mmproj" and report.launch_spec is not None:
                issue_labels.append(report.launch_spec.mmproj_path.name)
            else:
                issue_labels.append(f"{role_name}:{issue.code}")

    if issue_messages:
        return GuiRuntimeStatus(
            ready=False,
            headline="Check",
            detail=_format_doctor_status_detail(
                runtime_detail=describe_runtime_binding(binding),
                runtime_binary_path=effective_binary_path,
                models_dir=effective_models_dir,
                issue_messages=tuple(issue_messages),
            ),
            missing_items=_dedupe_strings(issue_labels),
            runtime_binary_path=effective_binary_path,
            models_dir=effective_models_dir,
            runtime_source=binding.source,
            runtime_release_tag=binding.release_tag,
            runtime_variant_id=binding.variant_id,
        )

    return GuiRuntimeStatus(
        ready=True,
        headline="Ready",
        detail=_format_doctor_status_detail(
            runtime_detail=describe_runtime_binding(binding),
            runtime_binary_path=effective_binary_path,
            models_dir=effective_models_dir,
            issue_messages=(),
        ),
        missing_items=(),
        runtime_binary_path=effective_binary_path,
        models_dir=effective_models_dir,
        runtime_source=binding.source,
        runtime_release_tag=binding.release_tag,
        runtime_variant_id=binding.variant_id,
    )


def _raise_if_gui_check_cancelled(
    cancel_event: threading.Event | None,
    *,
    stage: str,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError(f"GUI runtime check cancelled during {stage}")


def derive_primary_action(state: GuiScreenState) -> GuiPrimaryAction:
    return GuiPrimaryAction(
        kind="convert",
        label="Run",
        enabled=(
            state.runtime_status.ready
            and state.input_sup is not None
            and state.output_srt is not None
        ),
    )


def derive_setup_action(state: GuiScreenState) -> GuiPrimaryAction:
    return GuiPrimaryAction(
        kind="setup",
        label="Set Up",
        enabled=True,
    )


def build_setup_request(*, min_pixels: int = 32768) -> SetupRequest:
    return build_setup_request_for_variant(
        min_pixels=min_pixels,
        runtime_variant="auto",
        install_prerequisites=False,
    )


def build_setup_request_for_variant(
    *,
    min_pixels: int = 32768,
    runtime_variant: str = "auto",
    install_prerequisites: bool = False,
) -> SetupRequest:
    models_dir = gui_managed_models_dir()
    gguf_dir = resolve_local_model_path(DEFAULT_GGUF_MODEL_ID, models_dir=models_dir)
    base_mmproj = (gguf_dir / DEFAULT_GGUF_MMPROJ_FILENAME).resolve()
    return SetupRequest(
        models_dir=models_dir,
        derived_mmproj_output_path=resolve_derived_mmproj_output_path(
            base_mmproj=base_mmproj,
            models_dir=models_dir,
            min_pixels=min_pixels,
        ),
        bootstrap_managed_runtime=(os.name == "nt"),
        runtime_variant=runtime_variant,
        install_prerequisites=install_prerequisites,
    )


def build_fast_convert_request(
    *,
    input_sup: Path,
    output_srt: Path,
    enable_furigana_mask: bool,
    runtime_status: GuiRuntimeStatus,
) -> ConvertRequest:
    return ConvertRequest(
        input_sup=input_sup.expanduser().resolve(),
        output_srt=output_srt.expanduser().resolve(),
        engine="llama-server",
        ocr_mode="fast",
        enable_furigana_mask=enable_furigana_mask,
        corrector="off",
        models_dir=runtime_status.models_dir,
        runtime_binary_path=runtime_status.runtime_binary_path,
    )


def _format_runtime_status_detail(
    *,
    runtime_detail: str,
    runtime_binary_path: Path | None,
    models_dir: Path,
    missing_items: tuple[str, ...],
    issue_messages: tuple[str, ...],
) -> str:
    lines: list[str] = []
    if missing_items:
        lines.append("Missing components:")
        lines.extend(f"- {item}" for item in missing_items)
        lines.append("")
    lines.append("Runtime:")
    lines.append(f"- {runtime_detail}")
    if runtime_binary_path is not None:
        lines.append(f"- Binary: {_display_path(runtime_binary_path)}")
    lines.append("Models:")
    lines.append(f"- Root: {_display_path(models_dir)}")
    if issue_messages:
        lines.extend(
            [
                "",
                "Issues:",
                *[f"- {message}" for message in issue_messages],
            ]
        )
    return "\n".join(lines)


def format_runtime_facts(
    *,
    status: GuiRuntimeStatus,
    selected_variant: str | None = None,
) -> str:
    lines: list[str] = []
    if selected_variant:
        lines.append(f"Target: {selected_variant}")
    if status.runtime_source == MANAGED_RUNTIME_SOURCE:
        active = "Managed runtime"
        if status.runtime_release_tag and status.runtime_variant_id:
            active = f"Managed: {status.runtime_release_tag} [{status.runtime_variant_id}]"
        elif status.runtime_release_tag:
            active = f"Managed: {status.runtime_release_tag}"
        elif status.runtime_variant_id:
            active = f"Managed: [{status.runtime_variant_id}]"
        lines.append(active)
    elif status.runtime_source == EXTERNAL_RUNTIME_SOURCE:
        lines.append("Active: External runtime")
    elif status.runtime_source == OVERRIDE_RUNTIME_SOURCE:
        lines.append("Active: Configured runtime")
    if status.runtime_binary_path is not None:
        lines.append(f"Path: {_display_path(status.runtime_binary_path)}")
    return "\n".join(lines)


def _format_doctor_status_detail(
    *,
    runtime_detail: str,
    runtime_binary_path: Path | None,
    models_dir: Path,
    issue_messages: tuple[str, ...],
) -> str:
    lines = [
        "Runtime test passed." if not issue_messages else "Runtime test needs attention.",
        "",
        "Runtime:",
        f"- {runtime_detail}",
    ]
    if runtime_binary_path is not None:
        lines.append(f"- Binary: {_display_path(runtime_binary_path)}")
    lines.extend(
        [
            "Models:",
            f"- Root: {_display_path(models_dir)}",
        ]
    )
    if issue_messages:
        lines.extend(
            [
                "",
                "Issues:",
                *[f"- {message}" for message in issue_messages],
            ]
        )
    return "\n".join(lines)


def _format_runtime_binding_summary(status: GuiRuntimeStatus) -> str:
    if status.runtime_source == MANAGED_RUNTIME_SOURCE:
        if status.runtime_release_tag and status.runtime_variant_id:
            return f"Managed {status.runtime_release_tag} [{status.runtime_variant_id}]"
        if status.runtime_release_tag:
            return f"Managed {status.runtime_release_tag}"
        if status.runtime_variant_id:
            return f"Managed [{status.runtime_variant_id}]"
        return "Managed runtime"
    if status.runtime_source == EXTERNAL_RUNTIME_SOURCE:
        return "External runtime"
    if status.runtime_source == OVERRIDE_RUNTIME_SOURCE:
        return "Configured runtime"
    if status.runtime_binary_path is not None:
        return "Runtime found"
    return "Runtime missing"


def _format_missing_items_summary(missing_items: tuple[str, ...]) -> str:
    first = missing_items[0]
    if len(missing_items) == 1:
        return f"missing {first}"
    return f"missing {first} +{len(missing_items) - 1}"


def _compact_status_detail(detail: str) -> str:
    if not detail:
        return ""
    lines = [line.strip() for line in detail.splitlines() if line.strip()]
    if not lines:
        return ""
    preferred = [
        line
        for line in lines
        if line not in {"Runtime test passed.", "Runtime test needs attention.", "Missing components:", "Runtime:", "Models:", "Issues:"}
        and not line.startswith("- Binary:")
        and not line.startswith("- Path:")
        and not line.startswith("- Root:")
        and not line.startswith("- Managed ")
        and not line.startswith("- External ")
        and not line.startswith("- Configured ")
    ]
    source = preferred or lines
    first = source[0]
    return first[2:] if first.startswith("- ") else first


def format_setup_summary(
    *,
    status: GuiRuntimeStatus,
    selected_variant: str | None = None,
) -> str:
    parts: list[str] = ["Done" if status.ready else "Needed"]
    binding_summary = _format_runtime_binding_summary(status)
    if binding_summary:
        parts.append(binding_summary)
    if selected_variant:
        parts.append(f"target {selected_variant}")
    if not status.ready and status.missing_items:
        parts.append(_format_missing_items_summary(status.missing_items))
    return " | ".join(parts)


def format_check_summary(*, state: str, detail: str = "") -> str:
    summary = {
        "idle": "Not tested",
        "busy": "Testing",
        "ok": "Passed",
        "fail": "Failed",
    }.get(state, "Not tested")
    compact_detail = _compact_status_detail(detail)
    if compact_detail and state in {"busy", "fail"}:
        return f"{summary} | {compact_detail}"
    return summary


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    text = str(resolved)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        local_app_data_path = str(Path(local_app_data).expanduser().resolve())
        if text.startswith(local_app_data_path):
            return "%LOCALAPPDATA%" + text[len(local_app_data_path):]
    home_path = str(Path.home().resolve())
    if text.startswith(home_path):
        return "~" + text[len(home_path):]
    return text


def _probe_runtime_validation_messages(
    *,
    binding_source: str,
    binary_path: Path | None,
) -> tuple[str, ...]:
    if binding_source == "missing" or binary_path is None or not binary_path.exists():
        return ()
    try:
        validate_llama_server_binary(binary_path, timeout=5)
    except RuntimeError as exc:
        return _summarize_runtime_validation_message(str(exc))
    return ()


def _load_persisted_validation_messages(
    *,
    binding_source: str,
    binary_path: Path | None,
) -> tuple[str, ...]:
    if binding_source != MANAGED_RUNTIME_SOURCE or binary_path is None or not binary_path.exists():
        return ()
    state = load_managed_runtime_state()
    if state is None:
        return ()
    if state.binary_path.resolve() != binary_path.resolve():
        return ()
    return ()


def _summarize_runtime_validation_message(message: str) -> tuple[str, ...]:
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        return ("Runtime startup validation failed.",)
    summarized: list[str] = []
    for line in lines:
        if line.startswith("Binary:"):
            continue
        if line == "managed llama.cpp runtime failed startup validation.":
            summarized.append("Runtime startup validation failed.")
            continue
        summarized.append(line)
    return tuple(summarized) if summarized else ("Runtime startup validation failed.",)
