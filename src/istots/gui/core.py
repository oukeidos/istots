from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from istots.app.convert import ConvertRequest
from istots.app.setup import SetupRequest
from istots.llama_runtime import (
    DEFAULT_LLAMA_SERVER_HOST,
    LlamaServerOverrides,
    detect_llama_server_path,
    resolve_llama_server_role_assets,
    run_llama_server_doctor,
)
from istots.model_store import DEFAULT_GGUF_FILENAME, DEFAULT_GGUF_MMPROJ_FILENAME
from istots.ocr import LOCAL_PADDLE_CTX_SIZE


@dataclass(frozen=True)
class GuiRuntimeStatus:
    ready: bool
    headline: str
    detail: str
    missing_items: tuple[str, ...]


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

    binary_path = detect_llama_server_path(runtime_binary_path)
    if binary_path is None or not binary_path.exists():
        missing_items.append("llama-server")

    ocr_assets = resolve_llama_server_role_assets("ocr", models_dir=models_dir, min_pixels=min_pixels)
    fast_assets = resolve_llama_server_role_assets("ocr-fast", models_dir=models_dir, min_pixels=min_pixels)

    if not ocr_assets.model_path.exists():
        missing_items.append(DEFAULT_GGUF_FILENAME)
    if not ocr_assets.mmproj_path.exists():
        missing_items.append(DEFAULT_GGUF_MMPROJ_FILENAME)
    if not fast_assets.mmproj_path.exists():
        missing_items.append(fast_assets.mmproj_path.name)

    if missing_items:
        joined = ", ".join(dict.fromkeys(missing_items))
        return GuiRuntimeStatus(
            ready=False,
            headline="Setup",
            detail=joined,
            missing_items=_dedupe_strings(missing_items),
        )

    return GuiRuntimeStatus(
        ready=True,
        headline="Ready",
        detail="",
        missing_items=(),
    )


def run_gui_doctor_check(
    *,
    models_dir: Path | None = None,
    runtime_binary_path: Path | None = None,
    min_pixels: int = 32768,
    host: str = DEFAULT_LLAMA_SERVER_HOST,
) -> GuiRuntimeStatus:
    issue_labels: list[str] = []
    issue_messages: list[str] = []

    for role in ("ocr", "ocr-fast"):
        report = run_llama_server_doctor(
            role=role,
            models_dir=models_dir,
            min_pixels=min_pixels,
            explicit_binary_path=runtime_binary_path,
            host=host,
            overrides=_default_gui_doctor_overrides(),
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
            detail=" ".join(issue_messages),
            missing_items=_dedupe_strings(issue_labels),
        )

    return GuiRuntimeStatus(
        ready=True,
        headline="Ready",
        detail="OK",
        missing_items=(),
    )


def derive_primary_action(state: GuiScreenState) -> GuiPrimaryAction:
    if not state.runtime_status.ready:
        return GuiPrimaryAction(kind="setup", label="Setup", enabled=True)
    if state.input_sup is None or state.output_srt is None:
        return GuiPrimaryAction(kind="convert", label="Run", enabled=False)
    return GuiPrimaryAction(kind="convert", label="Run", enabled=True)


def build_setup_request() -> SetupRequest:
    return SetupRequest()


def build_fast_convert_request(
    *,
    input_sup: Path,
    output_srt: Path,
    enable_furigana_mask: bool,
) -> ConvertRequest:
    return ConvertRequest(
        input_sup=input_sup.expanduser().resolve(),
        output_srt=output_srt.expanduser().resolve(),
        engine="llama-server",
        ocr_mode="fast",
        enable_furigana_mask=enable_furigana_mask,
        corrector="off",
    )
