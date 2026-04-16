from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from istots import pipeline
from istots.corrector import (
    CorrectorConfig,
    CorrectorMode,
    GeminiConfigurationError,
    GeminiRequestFailedError,
)
from istots.detector import HybridDetectorRecord
from istots.ocr import (
    OCRBackendConfig,
    OCREngine,
    PaddleOCRVLRuntimeOverrides,
    Qwen35RuntimeOverrides,
)


def _generated_token(prefix: str, index: int) -> str:
    return f"{prefix.upper()}_{index:02d}"


def _generated_kanji(index: int) -> str:
    return chr(0x4E10 + index)


def _generated_family_pair(index: int) -> tuple[str, str]:
    base = index * 2
    return (_generated_kanji(base), _generated_kanji(base + 1))


def _generated_dialogue_text(speaker: str, line_index: int) -> str:
    return f"({speaker}{_generated_token('speaker', 0)}) {_generated_token('line', line_index)}"


def test_convert_sup_to_srt_releases_backend_on_success(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    input_sup.write_bytes(b"")

    class FakeBackend:
        instances: list["FakeBackend"] = []

        def __init__(self, **kwargs) -> None:
            self.closed = False
            FakeBackend.instances.append(self)

        def recognize_batch(self, images):
            return []

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](0)
        return iter([])

    def fake_write_srt(entries, path):
        path.write_text("", encoding="utf-8")

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", lambda config: FakeBackend())
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", fake_write_srt)

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        verbose=False,
    )

    assert result.written_count == 0
    assert FakeBackend.instances[0].closed


def test_convert_sup_to_srt_releases_backend_on_error(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    input_sup.write_bytes(b"")

    frame = SimpleNamespace(
        window_id=0,
        left=0,
        top=0,
        right=1,
        bottom=1,
        start=timedelta(milliseconds=0),
        end=timedelta(milliseconds=10),
        image=Image.new("RGB", (2, 2), "white"),
    )

    class FakeBackend:
        instances: list["FakeBackend"] = []

        def __init__(self, **kwargs) -> None:
            self.closed = False
            FakeBackend.instances.append(self)

        def recognize_batch(self, images):
            raise RuntimeError("boom")

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](1)
        return iter([frame])

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", lambda config: FakeBackend())
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", lambda entries, path: None)

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.convert_sup_to_srt(
            input_sup=input_sup,
            output_srt=output_srt,
            verbose=False,
        )

    assert FakeBackend.instances[0].closed


def test_convert_sup_to_srt_applies_furigana_mask_when_enabled(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    input_sup.write_bytes(b"")

    original = Image.new("RGB", (2, 2), "white")
    masked = Image.new("RGB", (2, 2), "black")
    frame = SimpleNamespace(
        window_id=0,
        left=0,
        top=0,
        right=1,
        bottom=1,
        start=timedelta(milliseconds=0),
        end=timedelta(milliseconds=10),
        image=original,
    )

    class FakeBackend:
        captured: list[Image.Image] = []

        def __init__(self, **kwargs) -> None:
            return None

        def recognize_batch(self, images):
            FakeBackend.captured.extend(images)
            return [""]

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](1)
        return iter([frame])

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", lambda config: FakeBackend())
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", lambda entries, path: None)
    monkeypatch.setattr(
        pipeline,
        "build_furigana_masks",
        lambda images: [SimpleNamespace(image=masked) for _ in images],
    )

    pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        enable_furigana_mask=True,
        verbose=False,
    )

    assert FakeBackend.captured[0] is masked


def test_convert_sup_to_srt_skips_furigana_mask_when_disabled(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    input_sup.write_bytes(b"")

    original = Image.new("RGB", (2, 2), "white")
    frame = SimpleNamespace(
        window_id=0,
        left=0,
        top=0,
        right=1,
        bottom=1,
        start=timedelta(milliseconds=0),
        end=timedelta(milliseconds=10),
        image=original,
    )

    class FakeBackend:
        captured: list[Image.Image] = []

        def __init__(self, **kwargs) -> None:
            return None

        def recognize_batch(self, images):
            FakeBackend.captured.extend(images)
            return [""]

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](1)
        return iter([frame])

    calls = {"count": 0}

    def fake_build(images):
        calls["count"] += 1
        return [SimpleNamespace(image=image) for image in images]

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", lambda config: FakeBackend())
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", lambda entries, path: None)
    monkeypatch.setattr(pipeline, "build_furigana_masks", fake_build)

    pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        enable_furigana_mask=False,
        verbose=False,
    )

    assert calls["count"] == 0
    assert FakeBackend.captured[0] is original


def test_convert_sup_to_srt_builds_hf_backend_config(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    input_sup.write_bytes(b"")

    captured: list[OCRBackendConfig] = []

    class FakeBackend:
        def recognize_batch(self, images):
            return []

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](0)
        return iter([])

    def fake_create_backend(config: OCRBackendConfig):
        captured.append(config)
        return FakeBackend()

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", fake_create_backend)
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", lambda entries, path: path.write_text("", encoding="utf-8"))

    pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        model_id="org/model",
        max_new_tokens=99,
        local_files_only=False,
        verbose=False,
    )

    assert captured == [
        OCRBackendConfig(
            engine=OCREngine.HF,
            model_id="org/model",
            device="cpu",
            hf_dtype="auto",
            max_new_tokens=99,
            local_files_only=False,
            models_dir=None,
            role="ocr",
            profile="auto",
            binary_path=None,
            host="127.0.0.1",
            port=None,
            threads=None,
            threads_batch=None,
            gpu_layers=None,
            no_mmproj_offload=None,
            startup_timeout_sec=120.0,
        )
    ]


def test_convert_sup_to_srt_retries_backend_init_on_auto_gpu_failure(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    input_sup.write_bytes(b"")

    calls: list[OCRBackendConfig] = []

    class FakeBackend:
        def recognize_batch(self, images):
            return []

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](0)
        return iter([])

    def fake_create_backend(config: OCRBackendConfig):
        calls.append(config)
        if len(calls) == 1:
            raise RuntimeError("GPU init failed")
        return FakeBackend()

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "gpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", fake_create_backend)
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", lambda entries, path: path.write_text("", encoding="utf-8"))

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        hf_device="auto",
        verbose=False,
    )

    assert result.device_used == "cpu"
    assert calls == [
        OCRBackendConfig(
            engine=OCREngine.HF,
            model_id="PaddlePaddle/PaddleOCR-VL-1.5",
            device="gpu",
            hf_dtype="auto",
            max_new_tokens=256,
            local_files_only=True,
            models_dir=None,
            role="ocr",
            profile="auto",
            binary_path=None,
            host="127.0.0.1",
            port=None,
            threads=None,
            threads_batch=None,
            gpu_layers=None,
            no_mmproj_offload=None,
            startup_timeout_sec=120.0,
        ),
        OCRBackendConfig(
            engine=OCREngine.HF,
            model_id="PaddlePaddle/PaddleOCR-VL-1.5",
            device="cpu",
            hf_dtype="auto",
            max_new_tokens=256,
            local_files_only=True,
            models_dir=None,
            role="ocr",
            profile="auto",
            binary_path=None,
            host="127.0.0.1",
            port=None,
            threads=None,
            threads_batch=None,
            gpu_layers=None,
            no_mmproj_offload=None,
            startup_timeout_sec=120.0,
        ),
    ]


def test_convert_sup_to_srt_fast_mode_requires_llama_server(tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    input_sup.write_bytes(b"")

    with pytest.raises(ValueError, match="detector output requires the llama-server engine"):
        pipeline.convert_sup_to_srt(
            input_sup=input_sup,
            output_srt=output_srt,
            engine=OCREngine.HF,
            ocr_mode="fast",
            detector_output=tmp_path / "detector.jsonl",
            verbose=False,
        )


def test_convert_sup_to_srt_fast_mode_partitions_rows_and_restores_order(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    input_sup.write_bytes(b"")

    frames = [
        SimpleNamespace(
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=10),
            image=Image.new("RGB", (20, 2), "white"),
        ),
        SimpleNamespace(
            window_id=0,
            left=0,
            top=0,
            right=2,
            bottom=20,
            start=timedelta(milliseconds=10),
            end=timedelta(milliseconds=20),
            image=Image.new("RGB", (2, 20), "white"),
        ),
        SimpleNamespace(
            window_id=0,
            left=0,
            top=0,
            right=30,
            bottom=3,
            start=timedelta(milliseconds=20),
            end=timedelta(milliseconds=30),
            image=Image.new("RGB", (30, 3), "white"),
        ),
    ]

    created_roles: list[str] = []
    branch_calls: list[tuple[str, list[tuple[int, int]]]] = []
    closed_roles: list[str] = []
    written_entries = []
    live_count = 0
    max_live_count = 0

    class FakeBackend:
        def __init__(self, role: str) -> None:
            nonlocal live_count, max_live_count
            self.role = role
            self.calls = 0
            live_count += 1
            max_live_count = max(max_live_count, live_count)

        def recognize_batch(self, images):
            branch_calls.append((self.role, [image.size for image in images]))
            self.calls += 1
            prefix = "fast" if self.role == "ocr-fast" else "default"
            return [f"{prefix}-{index}" for index in range(self.calls, self.calls + len(images))]

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            nonlocal live_count
            closed_roles.append(self.role)
            live_count -= 1

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](len(frames))
        return iter(frames)

    def fake_create_backend(config: OCRBackendConfig):
        created_roles.append(config.role)
        return FakeBackend(config.role)

    def fake_write_srt(entries, path):
        written_entries.extend(entries)
        path.write_text("", encoding="utf-8")

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", fake_create_backend)
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", fake_write_srt)

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        engine=OCREngine.LLAMA_SERVER,
        ocr_mode="fast",
        srt_policy="overlap",
        verbose=False,
    )

    assert result.processed_count == 3
    assert created_roles == ["ocr-fast", "ocr"]
    assert branch_calls == [
        ("ocr-fast", [(20, 2)]),
        ("ocr-fast", [(30, 3)]),
        ("ocr", [(2, 20)]),
    ]
    assert [entry.text for entry in written_entries] == ["fast-1", "default-1", "fast-2"]
    assert closed_roles == ["ocr-fast", "ocr"]
    assert max_live_count == 1


def test_convert_sup_to_srt_fast_mode_uses_hf_min_pixels_on_non_tall_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    input_sup.write_bytes(b"")

    frames = [
        SimpleNamespace(
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=10),
            image=Image.new("RGB", (20, 2), "white"),
        ),
        SimpleNamespace(
            window_id=0,
            left=0,
            top=0,
            right=2,
            bottom=20,
            start=timedelta(milliseconds=10),
            end=timedelta(milliseconds=20),
            image=Image.new("RGB", (2, 20), "white"),
        ),
    ]

    created_configs: list[OCRBackendConfig] = []
    written_entries = []

    class FakeBackend:
        def __init__(self, role: str) -> None:
            self.role = role

        def recognize_batch(self, images):
            if self.role == "ocr-fast":
                return ["FAST-WIDE"]
            if self.role == "ocr":
                return ["DEFAULT-TALL"]
            raise AssertionError(f"unexpected role {self.role}")

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](len(frames))
        return iter(frames)

    def fake_create_backend(config: OCRBackendConfig):
        created_configs.append(config)
        return FakeBackend(config.role)

    def fake_write_srt(entries, path):
        written_entries.extend(entries)
        path.write_text("", encoding="utf-8")

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", fake_create_backend)
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", fake_write_srt)

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        engine=OCREngine.HF,
        hf_device="cpu",
        ocr_mode="fast",
        srt_policy="overlap",
        verbose=False,
    )

    assert result.processed_count == 2
    assert created_configs[0].role == "ocr-fast"
    assert created_configs[0].hf_min_pixels == pipeline.HF_FAST_MIN_PIXELS
    assert created_configs[1].role == "ocr"
    assert created_configs[1].hf_min_pixels is None
    assert [entry.text for entry in written_entries] == ["FAST-WIDE", "DEFAULT-TALL"]


def test_convert_sup_to_srt_writes_hybrid_detector_manifest(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    detector_output = tmp_path / "detector.jsonl"
    input_sup.write_bytes(b"")

    frames = [
        SimpleNamespace(
            raw_index=10,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=10),
            image=Image.new("RGB", (20, 2), (255, 255, 255)),
        ),
        SimpleNamespace(
            raw_index=11,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=10),
            end=timedelta(milliseconds=20),
            image=Image.new("RGB", (20, 2), (230, 230, 230)),
        ),
        SimpleNamespace(
            raw_index=12,
            window_id=0,
            left=0,
            top=0,
            right=2,
            bottom=20,
            start=timedelta(milliseconds=20),
            end=timedelta(milliseconds=30),
            image=Image.new("RGB", (2, 20), (200, 200, 200)),
        ),
    ]

    created_roles: list[str] = []
    closed_roles: list[str] = []
    live_count = 0
    max_live_count = 0

    class FakeBackend:
        def __init__(self, role: str) -> None:
            nonlocal live_count, max_live_count
            self.role = role
            live_count += 1
            max_live_count = max(max_live_count, live_count)
            self.calls = 0

        def recognize_batch(self, images):
            self.calls += 1
            if self.role == "ocr":
                return [["CTRL"], ["BASE-WIDE"], ["BASE-TALL"]][self.calls - 1]
            if self.role == "ocr-fast":
                return [["CTRL"], ["ALT-WIDE"]][self.calls - 1]
            if self.role == "detector":
                return ["ALT-TALL"]
            raise AssertionError(f"unexpected role {self.role}")

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            nonlocal live_count
            closed_roles.append(self.role)
            live_count -= 1

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](len(frames))
        return iter(frames)

    def fake_create_backend(config: OCRBackendConfig):
        created_roles.append(config.role)
        return FakeBackend(config.role)

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", fake_create_backend)
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", lambda entries, path: path.write_text("", encoding="utf-8"))

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        engine=OCREngine.LLAMA_SERVER,
        detector_output=detector_output,
        srt_policy="overlap",
        verbose=False,
    )

    manifest = [json.loads(line) for line in detector_output.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result.processed_count == 3
    assert result.detector_record_count == 2
    assert created_roles == ["ocr", "ocr-fast", "detector"]
    assert closed_roles == ["ocr", "ocr-fast", "detector"]
    assert max_live_count == 1
    assert [row["detector_branch"] for row in manifest] == [
        "alternate_read_non_tall",
        "repeat_drift_tall",
    ]
    assert [row["option_role"] for row in manifest] == ["ocr-fast", "detector"]
    assert [row["baseline_text"] for row in manifest] == ["BASE-WIDE", "BASE-TALL"]
    assert [row["option_text"] for row in manifest] == ["ALT-WIDE", "ALT-TALL"]
    assert [row["source_tags"] for row in manifest] == [["hybrid_detector"], ["hybrid_detector"]]
    assert [row["alternate_source_kind"] for row in manifest] == ["min32768", "temp0_repeat"]
    assert [row["diff_label"] for row in manifest] == [
        "meaningful_difference",
        "meaningful_difference",
    ]


def test_convert_sup_to_srt_detector_family_addon_appends_agreement_rows(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    detector_output = tmp_path / "detector.jsonl"
    input_sup.write_bytes(b"")

    frames = [
        SimpleNamespace(
            raw_index=10,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=10),
            image=Image.new("RGB", (20, 2), (255, 255, 255)),
        ),
        SimpleNamespace(
            raw_index=11,
            window_id=0,
            left=0,
            top=0,
            right=2,
            bottom=20,
            start=timedelta(milliseconds=10),
            end=timedelta(milliseconds=20),
            image=Image.new("RGB", (2, 20), (235, 235, 235)),
        ),
        SimpleNamespace(
            raw_index=12,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=20),
            end=timedelta(milliseconds=30),
            image=Image.new("RGB", (20, 2), (215, 215, 215)),
        ),
        SimpleNamespace(
            raw_index=13,
            window_id=0,
            left=0,
            top=0,
            right=2,
            bottom=20,
            start=timedelta(milliseconds=30),
            end=timedelta(milliseconds=40),
            image=Image.new("RGB", (2, 20), (195, 195, 195)),
        ),
        SimpleNamespace(
            raw_index=14,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=40),
            end=timedelta(milliseconds=50),
            image=Image.new("RGB", (20, 2), (175, 175, 175)),
        ),
    ]
    family_primary, family_alternate = _generated_family_pair(0)
    unrelated_speaker = _generated_kanji(10)
    family_name = "".join(sorted(family_primary + family_alternate))

    class FakeBackend:
        def __init__(self, role: str) -> None:
            self.role = role
            self.calls = 0

        def recognize_batch(self, images):
            self.calls += 1
            if self.role == "ocr":
                return [[
                    _generated_dialogue_text(family_primary, 0),
                    _generated_dialogue_text(family_primary, 1),
                    _generated_dialogue_text(family_primary, 2),
                    _generated_dialogue_text(family_alternate, 3),
                    _generated_dialogue_text(unrelated_speaker, 0),
                ][self.calls - 1]]
            if self.role == "ocr-fast":
                return [[
                    _generated_dialogue_text(family_alternate, 0),
                    _generated_dialogue_text(family_primary, 2),
                    _generated_dialogue_text(unrelated_speaker, 0),
                ][self.calls - 1]]
            if self.role == "detector":
                return [[
                    _generated_dialogue_text(family_alternate, 1),
                    _generated_dialogue_text(family_alternate, 3),
                ][self.calls - 1]]
            raise AssertionError(f"unexpected role {self.role}")

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](len(frames))
        return iter(frames)

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", lambda config: FakeBackend(config.role))
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", lambda entries, path: path.write_text("", encoding="utf-8"))

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        engine=OCREngine.LLAMA_SERVER,
        detector_output=detector_output,
        detector_family_addon=True,
        srt_policy="overlap",
        verbose=False,
    )

    manifest = [json.loads(line) for line in detector_output.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result.detector_record_count == 4
    assert [row["detector_branch"] for row in manifest] == [
        "alternate_read_non_tall",
        "repeat_drift_tall",
        "dominant_family_addon",
        "dominant_family_addon",
    ]
    assert manifest[2]["baseline_text"] == _generated_dialogue_text(family_primary, 2)
    assert manifest[2]["option_text"] == _generated_dialogue_text(family_alternate, 2)
    assert manifest[2]["alternate_source_kind"] == "family_pair_swap"
    assert manifest[2]["dominant_family"] == family_name
    assert manifest[2]["family_current_char"] == family_primary
    assert manifest[2]["family_alternate_char"] == family_alternate
    assert manifest[2]["source_tags"] == ["dominant_family_addon"]
    assert manifest[2]["family_support_rows"] == 2
    assert manifest[2]["family_pure_rows"] == 2
    assert manifest[2]["family_mixed_rows"] == 0
    assert manifest[2]["family_agreement_rows"] == 2
    assert manifest[3]["baseline_text"] == _generated_dialogue_text(family_alternate, 3)
    assert manifest[3]["option_text"] == _generated_dialogue_text(family_primary, 3)
    assert manifest[3]["dominant_family"] == family_name


def test_convert_sup_to_srt_wider_detector_merges_p2_surface_rows(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    detector_output = tmp_path / "detector.jsonl"
    input_sup.write_bytes(b"")

    frames = [
        SimpleNamespace(
            raw_index=10,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=10),
            image=Image.new("RGB", (20, 2), (255, 255, 255)),
        ),
        SimpleNamespace(
            raw_index=11,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=10),
            end=timedelta(milliseconds=20),
            image=Image.new("RGB", (20, 2), (235, 235, 235)),
        ),
        SimpleNamespace(
            raw_index=12,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=20),
            end=timedelta(milliseconds=30),
            image=Image.new("RGB", (20, 2), (215, 215, 215)),
        ),
    ]

    detector_instance_count = 0
    created_roles: list[str] = []
    baseline_0 = _generated_token("base", 0)
    baseline_1 = _generated_token("base", 1)
    baseline_2 = _generated_token("base", 2)
    alt_s1_0 = _generated_token("alt_s1", 0)
    alt_p2_0 = _generated_token("alt_p2", 0)
    alt_p2_2 = _generated_token("alt_p2", 2)

    class FakeBackend:
        def __init__(self, role: str, detector_instance: int | None = None) -> None:
            self.role = role
            self.detector_instance = detector_instance
            self.calls = 0

        def recognize_batch(self, images):
            self.calls += 1
            if self.role == "ocr":
                return [[baseline_0], [baseline_1], [baseline_2]][self.calls - 1]
            if self.role == "ocr-fast":
                return [[alt_s1_0], [baseline_1], [baseline_2]][self.calls - 1]
            if self.role == "detector" and self.detector_instance == 1:
                return [[alt_p2_0], [baseline_1], [alt_p2_2]][self.calls - 1]
            raise AssertionError(f"unexpected role {self.role}#{self.detector_instance}")

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](len(frames))
        return iter(frames)

    def fake_create_backend(config: OCRBackendConfig):
        nonlocal detector_instance_count
        created_roles.append(config.role)
        if config.role == "detector":
            detector_instance_count += 1
            return FakeBackend(config.role, detector_instance=detector_instance_count)
        return FakeBackend(config.role)

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", fake_create_backend)
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", lambda entries, path: path.write_text("", encoding="utf-8"))

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        engine=OCREngine.LLAMA_SERVER,
        detector_output=detector_output,
        detector_mode="wider",
        srt_policy="overlap",
        verbose=False,
    )

    manifest = [json.loads(line) for line in detector_output.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result.detector_record_count == 2
    assert created_roles == ["ocr", "ocr-fast", "detector"]
    assert [row["detector_branch"] for row in manifest] == [
        "alternate_read_non_tall",
        "p2_meaningful_temp0",
    ]
    assert manifest[0]["baseline_text"] == baseline_0
    assert manifest[0]["option_text"] == alt_s1_0
    assert manifest[0]["source_tags"] == ["hybrid_detector", "p2_meaningful_temp0"]
    assert manifest[0]["alternate_source_kind"] == "min32768"
    assert manifest[1]["baseline_text"] == baseline_2
    assert manifest[1]["option_text"] == alt_p2_2
    assert manifest[1]["source_tags"] == ["p2_meaningful_temp0"]
    assert manifest[1]["alternate_source_kind"] == "temp0_repeat"


def test_convert_sup_to_srt_detector_family_addon_can_attach_to_wider_surface(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    detector_output = tmp_path / "detector.jsonl"
    input_sup.write_bytes(b"")

    frames = [
        SimpleNamespace(
            raw_index=10 + index,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=index * 10),
            end=timedelta(milliseconds=(index + 1) * 10),
            image=Image.new("RGB", (20, 2), (index, index, index)),
        )
        for index in range(4)
    ]

    detector_instance_count = 0
    family_primary, family_alternate = _generated_family_pair(0)
    unrelated_speaker = _generated_kanji(10)
    family_name = "".join(sorted(family_primary + family_alternate))

    class FakeBackend:
        def __init__(self, role: str, detector_instance: int | None = None) -> None:
            self.role = role
            self.detector_instance = detector_instance
            self.calls = 0

        def recognize_batch(self, images):
            self.calls += 1
            if self.role == "ocr":
                return [[
                    _generated_dialogue_text(family_primary, 0),
                    _generated_dialogue_text(family_primary, 1),
                    _generated_dialogue_text(family_primary, 2),
                    _generated_dialogue_text(unrelated_speaker, 0),
                ][self.calls - 1]]
            if self.role == "ocr-fast":
                return [[
                    _generated_dialogue_text(family_primary, 0),
                    _generated_dialogue_text(family_primary, 1),
                    _generated_dialogue_text(family_primary, 2),
                    _generated_dialogue_text(unrelated_speaker, 0),
                ][self.calls - 1]]
            if self.role == "detector" and self.detector_instance == 1:
                return [[
                    _generated_dialogue_text(family_alternate, 0),
                    _generated_dialogue_text(family_alternate, 1),
                    _generated_dialogue_text(family_primary, 2),
                    _generated_dialogue_text(unrelated_speaker, 0),
                ][self.calls - 1]]
            raise AssertionError(f"unexpected role {self.role}#{self.detector_instance}")

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](len(frames))
        return iter(frames)

    def fake_create_backend(config: OCRBackendConfig):
        nonlocal detector_instance_count
        if config.role == "detector":
            detector_instance_count += 1
            return FakeBackend(config.role, detector_instance=detector_instance_count)
        return FakeBackend(config.role)

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", fake_create_backend)
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", lambda entries, path: path.write_text("", encoding="utf-8"))

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        engine=OCREngine.LLAMA_SERVER,
        detector_output=detector_output,
        detector_mode="wider",
        detector_family_addon=True,
        srt_policy="overlap",
        verbose=False,
    )

    manifest = [json.loads(line) for line in detector_output.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result.detector_record_count == 3
    assert [row["detector_branch"] for row in manifest] == [
        "p2_meaningful_temp0",
        "p2_meaningful_temp0",
        "dominant_family_addon",
    ]
    assert manifest[0]["source_tags"] == ["p2_meaningful_temp0"]
    assert manifest[1]["source_tags"] == ["p2_meaningful_temp0"]
    assert manifest[2]["baseline_text"] == _generated_dialogue_text(family_primary, 2)
    assert manifest[2]["option_text"] == _generated_dialogue_text(family_alternate, 2)
    assert manifest[2]["dominant_family"] == family_name
    assert manifest[2]["source_tags"] == ["dominant_family_addon"]
    assert manifest[2]["family_support_rows"] == 2
    assert manifest[2]["family_pure_rows"] == 2
    assert manifest[2]["family_mixed_rows"] == 0
    assert manifest[2]["family_agreement_rows"] == 1


def test_convert_sup_to_srt_wider_detector_reuses_exact_duplicate_images(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    detector_output = tmp_path / "detector.jsonl"
    input_sup.write_bytes(b"")

    frames = [
        SimpleNamespace(
            raw_index=10,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=10),
            image=Image.new("RGB", (20, 2), "white"),
        ),
        SimpleNamespace(
            raw_index=11,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=10),
            end=timedelta(milliseconds=20),
            image=Image.new("RGB", (20, 2), "white"),
        ),
        SimpleNamespace(
            raw_index=12,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=20),
            end=timedelta(milliseconds=30),
            image=Image.new("RGB", (20, 2), (210, 210, 210)),
        ),
        SimpleNamespace(
            raw_index=13,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=30),
            end=timedelta(milliseconds=40),
            image=Image.new("RGB", (20, 2), (180, 180, 180)),
        ),
    ]

    detector_instance_count = 0
    created_roles: list[str] = []
    role_calls: dict[str, int] = {}
    baseline_dup = _generated_token("base_dup", 0)
    baseline_2 = _generated_token("base", 2)
    baseline_3 = _generated_token("base", 3)
    alt_dup = _generated_token("alt_dup", 0)

    class FakeBackend:
        def __init__(self, role: str, detector_instance: int | None = None) -> None:
            self.role = role
            self.detector_instance = detector_instance
            self.calls = 0

        def recognize_batch(self, images):
            self.calls += 1
            key = self.role if self.detector_instance is None else f"{self.role}#{self.detector_instance}"
            role_calls[key] = role_calls.get(key, 0) + 1
            if self.role == "ocr":
                return [[baseline_dup], [baseline_2], [baseline_3]][self.calls - 1]
            if self.role == "ocr-fast":
                return [[baseline_dup], [baseline_2], [baseline_3]][self.calls - 1]
            if self.role == "detector" and self.detector_instance == 1:
                return [[alt_dup], [baseline_2], [baseline_3]][self.calls - 1]
            raise AssertionError(f"unexpected role {self.role}#{self.detector_instance}")

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](len(frames))
        return iter(frames)

    def fake_create_backend(config: OCRBackendConfig):
        nonlocal detector_instance_count
        created_roles.append(config.role)
        if config.role == "detector":
            detector_instance_count += 1
            return FakeBackend(config.role, detector_instance=detector_instance_count)
        return FakeBackend(config.role)

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", fake_create_backend)
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", lambda entries, path: path.write_text("", encoding="utf-8"))

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        engine=OCREngine.LLAMA_SERVER,
        detector_output=detector_output,
        detector_mode="wider",
        srt_policy="overlap",
        verbose=False,
    )

    manifest = [json.loads(line) for line in detector_output.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result.detector_record_count == 2
    assert created_roles == ["ocr", "ocr-fast", "detector"]
    assert role_calls == {"ocr": 3, "ocr-fast": 3, "detector#1": 3}
    assert [row["detector_branch"] for row in manifest] == [
        "p2_meaningful_temp0",
        "p2_meaningful_temp0",
    ]
    assert [row["baseline_text"] for row in manifest] == [baseline_dup, baseline_dup]
    assert [row["option_text"] for row in manifest] == [alt_dup, alt_dup]
    assert [row["source_tags"] for row in manifest] == [
        ["p2_meaningful_temp0"],
        ["p2_meaningful_temp0"],
    ]


def test_convert_sup_to_srt_wider_detector_family_addon_reuses_exact_duplicate_images(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    detector_output = tmp_path / "detector.jsonl"
    input_sup.write_bytes(b"")

    frames = [
        SimpleNamespace(
            raw_index=10,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=10),
            image=Image.new("RGB", (20, 2), "white"),
        ),
        SimpleNamespace(
            raw_index=11,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=10),
            end=timedelta(milliseconds=20),
            image=Image.new("RGB", (20, 2), "white"),
        ),
        SimpleNamespace(
            raw_index=12,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=20),
            end=timedelta(milliseconds=30),
            image=Image.new("RGB", (20, 2), (210, 210, 210)),
        ),
        SimpleNamespace(
            raw_index=13,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=30),
            end=timedelta(milliseconds=40),
            image=Image.new("RGB", (20, 2), (180, 180, 180)),
        ),
    ]

    detector_instance_count = 0
    created_roles: list[str] = []
    role_calls: dict[str, int] = {}
    family_primary, family_alternate = _generated_family_pair(0)
    unrelated_speaker = _generated_kanji(10)
    family_name = "".join(sorted(family_primary + family_alternate))

    class FakeBackend:
        def __init__(self, role: str, detector_instance: int | None = None) -> None:
            self.role = role
            self.detector_instance = detector_instance
            self.calls = 0

        def recognize_batch(self, images):
            self.calls += 1
            key = self.role if self.detector_instance is None else f"{self.role}#{self.detector_instance}"
            role_calls[key] = role_calls.get(key, 0) + 1
            if self.role == "ocr":
                return [[
                    _generated_dialogue_text(family_primary, 0),
                    _generated_dialogue_text(family_primary, 2),
                    _generated_dialogue_text(unrelated_speaker, 0),
                ][self.calls - 1]]
            if self.role == "ocr-fast":
                return [[
                    _generated_dialogue_text(family_primary, 0),
                    _generated_dialogue_text(family_primary, 2),
                    _generated_dialogue_text(unrelated_speaker, 0),
                ][self.calls - 1]]
            if self.role == "detector" and self.detector_instance == 1:
                return [[
                    _generated_dialogue_text(family_alternate, 0),
                    _generated_dialogue_text(family_primary, 2),
                    _generated_dialogue_text(unrelated_speaker, 0),
                ][self.calls - 1]]
            raise AssertionError(f"unexpected role {self.role}#{self.detector_instance}")

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](len(frames))
        return iter(frames)

    def fake_create_backend(config: OCRBackendConfig):
        nonlocal detector_instance_count
        created_roles.append(config.role)
        if config.role == "detector":
            detector_instance_count += 1
            return FakeBackend(config.role, detector_instance=detector_instance_count)
        return FakeBackend(config.role)

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", fake_create_backend)
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", lambda entries, path: path.write_text("", encoding="utf-8"))

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        engine=OCREngine.LLAMA_SERVER,
        detector_output=detector_output,
        detector_mode="wider",
        detector_family_addon=True,
        srt_policy="overlap",
        verbose=False,
    )

    manifest = [json.loads(line) for line in detector_output.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result.detector_record_count == 3
    assert created_roles == ["ocr", "ocr-fast", "detector"]
    assert role_calls == {"ocr": 3, "ocr-fast": 3, "detector#1": 3}
    assert [row["detector_branch"] for row in manifest] == [
        "p2_meaningful_temp0",
        "p2_meaningful_temp0",
        "dominant_family_addon",
    ]
    assert [row["baseline_text"] for row in manifest[:2]] == [
        _generated_dialogue_text(family_primary, 0),
        _generated_dialogue_text(family_primary, 0),
    ]
    assert [row["option_text"] for row in manifest[:2]] == [
        _generated_dialogue_text(family_alternate, 0),
        _generated_dialogue_text(family_alternate, 0),
    ]
    assert manifest[2]["baseline_text"] == _generated_dialogue_text(family_primary, 2)
    assert manifest[2]["option_text"] == _generated_dialogue_text(family_alternate, 2)
    assert manifest[2]["source_tags"] == ["dominant_family_addon"]
    assert manifest[2]["family_support_rows"] == 2
    assert manifest[2]["family_pure_rows"] == 2
    assert manifest[2]["family_mixed_rows"] == 0
    assert manifest[2]["family_agreement_rows"] == 1
    assert manifest[2]["dominant_family"] == family_name


def test_select_dominant_kanji_family_ignores_non_kanji_pairs() -> None:
    records = [
        HybridDetectorRecord(
            index=0,
            raw_index=0,
            window_id=0,
            start_ms=0,
            end_ms=10,
            detector_branch="alternate_read_non_tall",
            shape="wide",
            ratio=0.1,
            option_role="ocr-fast",
            baseline_text="かなが",
            option_text="かなか",
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=0.2,
        ),
        HybridDetectorRecord(
            index=1,
            raw_index=1,
            window_id=0,
            start_ms=10,
            end_ms=20,
            detector_branch="repeat_drift_tall",
            shape="tall",
            ratio=3.0,
            option_role="detector",
            baseline_text="かなが",
            option_text="かなか",
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=0.2,
        ),
    ]

    assert (
        pipeline._select_dominant_kanji_family(
            prepared_inputs=[],
            baseline_texts=[],
            s1_detector_records=records,
        )
        is None
    )


def test_select_dominant_kanji_family_prefers_purer_row_level_candidate() -> None:
    family_primary, family_alternate = _generated_family_pair(0)
    secondary_primary, secondary_alternate = _generated_family_pair(1)
    tertiary_primary, tertiary_alternate = _generated_family_pair(2)
    preferred_family_name = "".join(sorted(family_primary + family_alternate))

    records = [
        HybridDetectorRecord(
            index=0,
            raw_index=0,
            window_id=0,
            start_ms=0,
            end_ms=10,
            detector_branch="alternate_read_non_tall",
            shape="wide",
            ratio=0.1,
            option_role="ocr-fast",
            baseline_text=family_primary,
            option_text=family_alternate,
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=1.0,
        ),
        HybridDetectorRecord(
            index=1,
            raw_index=1,
            window_id=0,
            start_ms=10,
            end_ms=20,
            detector_branch="alternate_read_non_tall",
            shape="wide",
            ratio=0.1,
            option_role="ocr-fast",
            baseline_text=family_primary,
            option_text=family_alternate,
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=1.0,
        ),
        HybridDetectorRecord(
            index=2,
            raw_index=2,
            window_id=0,
            start_ms=20,
            end_ms=30,
            detector_branch="alternate_read_non_tall",
            shape="wide",
            ratio=0.1,
            option_role="ocr-fast",
            baseline_text=secondary_primary,
            option_text=secondary_alternate,
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=1.0,
        ),
        HybridDetectorRecord(
            index=3,
            raw_index=3,
            window_id=0,
            start_ms=30,
            end_ms=40,
            detector_branch="alternate_read_non_tall",
            shape="wide",
            ratio=0.1,
            option_role="ocr-fast",
            baseline_text=f"{family_primary}/{secondary_primary}",
            option_text=f"{family_alternate}/{secondary_alternate}",
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=1.0,
        ),
        HybridDetectorRecord(
            index=4,
            raw_index=4,
            window_id=0,
            start_ms=40,
            end_ms=50,
            detector_branch="alternate_read_non_tall",
            shape="wide",
            ratio=0.1,
            option_role="ocr-fast",
            baseline_text=f"{secondary_primary}/{tertiary_primary}",
            option_text=f"{secondary_alternate}/{tertiary_alternate}",
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=1.0,
        ),
    ]
    prepared_inputs = [
        pipeline._PreparedOCRInput(
            index=index,
            frame=SimpleNamespace(raw_index=index, window_id=0, start=timedelta(), end=timedelta()),
            image=Image.new("RGB", (1, 1), "white"),
        )
        for index in range(7)
    ]
    baseline_texts = [
        family_primary,
        family_primary,
        secondary_primary,
        f"{family_primary}/{secondary_primary}",
        f"{secondary_primary}/{tertiary_primary}",
        f"{family_primary}{_generated_token('suffix', 0)}",
        f"{secondary_primary}{_generated_token('suffix', 0)}",
    ]

    selected = pipeline._select_dominant_kanji_family(
        prepared_inputs=prepared_inputs,
        baseline_texts=baseline_texts,
        s1_detector_records=records,
    )

    assert selected is not None
    assert selected.family == preferred_family_name
    assert selected.support_rows == 3
    assert selected.pure_rows == 2
    assert selected.mixed_rows == 1
    assert selected.agreement_rows == 1


def test_select_dominant_kanji_family_rejects_overly_broad_family() -> None:
    family_primary, family_alternate = _generated_family_pair(0)
    secondary_primary, secondary_alternate = _generated_family_pair(1)
    secondary_family_name = "".join(sorted(secondary_primary + secondary_alternate))

    records = [
        HybridDetectorRecord(
            index=0,
            raw_index=0,
            window_id=0,
            start_ms=0,
            end_ms=10,
            detector_branch="alternate_read_non_tall",
            shape="wide",
            ratio=0.1,
            option_role="ocr-fast",
            baseline_text=family_primary,
            option_text=family_alternate,
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=1.0,
        ),
        HybridDetectorRecord(
            index=1,
            raw_index=1,
            window_id=0,
            start_ms=10,
            end_ms=20,
            detector_branch="alternate_read_non_tall",
            shape="wide",
            ratio=0.1,
            option_role="ocr-fast",
            baseline_text=family_primary,
            option_text=family_alternate,
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=1.0,
        ),
        HybridDetectorRecord(
            index=2,
            raw_index=2,
            window_id=0,
            start_ms=20,
            end_ms=30,
            detector_branch="alternate_read_non_tall",
            shape="wide",
            ratio=0.1,
            option_role="ocr-fast",
            baseline_text=secondary_primary,
            option_text=secondary_alternate,
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=1.0,
        ),
        HybridDetectorRecord(
            index=3,
            raw_index=3,
            window_id=0,
            start_ms=30,
            end_ms=40,
            detector_branch="alternate_read_non_tall",
            shape="wide",
            ratio=0.1,
            option_role="ocr-fast",
            baseline_text=secondary_primary,
            option_text=secondary_alternate,
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=1.0,
        ),
    ]
    baseline_texts = [family_primary, family_primary, secondary_primary, secondary_primary]
    baseline_texts.extend([f"{family_primary}{_generated_token('suffix', 0)}"] * 21)
    baseline_texts.extend([f"{secondary_primary}{_generated_token('suffix', 0)}"] * 2)
    prepared_inputs = [
        pipeline._PreparedOCRInput(
            index=index,
            frame=SimpleNamespace(raw_index=index, window_id=0, start=timedelta(), end=timedelta()),
            image=Image.new("RGB", (1, 1), "white"),
        )
        for index in range(len(baseline_texts))
    ]

    selected = pipeline._select_dominant_kanji_family(
        prepared_inputs=prepared_inputs,
        baseline_texts=baseline_texts,
        s1_detector_records=records,
    )

    assert selected is not None
    assert selected.family == secondary_family_name
    assert selected.agreement_rows == 2


def test_convert_sup_to_srt_correction_requires_llama_server(tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    input_sup.write_bytes(b"")

    with pytest.raises(ValueError, match="correction requires the llama-server engine"):
        pipeline.convert_sup_to_srt(
            input_sup=input_sup,
            output_srt=output_srt,
            engine=OCREngine.HF,
            corrector_config=CorrectorConfig(mode=CorrectorMode.GEMINI),
            verbose=False,
        )


def test_convert_sup_to_srt_applies_local_conservative_correction(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    corrector_output = tmp_path / "corrected.jsonl"
    input_sup.write_bytes(b"")

    frames = [
        SimpleNamespace(
            raw_index=10,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=10),
            image=Image.new("RGB", (20, 2), "white"),
        )
    ]

    created_configs: list[OCRBackendConfig] = []
    closed_roles: list[str] = []
    written_entries = []
    live_count = 0
    max_live_count = 0

    class FakeBackend:
        def __init__(self, role: str) -> None:
            nonlocal live_count, max_live_count
            self.role = role
            live_count += 1
            max_live_count = max(max_live_count, live_count)

        def recognize_batch(self, images):
            if self.role == "ocr":
                return ["ABC"]
            if self.role == "ocr-fast":
                return ["ADC"]
            if self.role == "corrector":
                return ["AECZ"]
            raise AssertionError(f"unexpected role {self.role}")

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            nonlocal live_count
            closed_roles.append(self.role)
            live_count -= 1

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](len(frames))
        return iter(frames)

    def fake_create_backend(config: OCRBackendConfig):
        created_configs.append(config)
        return FakeBackend(config.role)

    def fake_write_srt(entries, path):
        written_entries.extend(entries)
        path.write_text("", encoding="utf-8")

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", fake_create_backend)
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", fake_write_srt)

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        engine=OCREngine.LLAMA_SERVER,
        paddle_runtime_overrides=PaddleOCRVLRuntimeOverrides(profile="cpu"),
        corrector_config=CorrectorConfig(
            mode=CorrectorMode.QWEN_LOCAL,
            output_path=corrector_output,
            local_model_path=tmp_path / "qwen.gguf",
            local_mmproj_path=tmp_path / "qwen-mmproj.gguf",
            local_runtime_overrides=Qwen35RuntimeOverrides(profile="cpu"),
        ),
        srt_policy="overlap",
        verbose=False,
    )

    manifest = [json.loads(line) for line in corrector_output.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result.detector_record_count == 1
    assert result.correction_record_count == 1
    assert result.correction_applied_count == 1
    assert [config.role for config in created_configs] == ["ocr", "ocr-fast", "corrector"]
    assert created_configs[2].prompt_text == (
        "Transcribe only the visible subtitle text in the image. Output only the text. "
        "Preserve line breaks. Do not explain."
    )
    assert created_configs[2].reasoning == "off"
    assert created_configs[2].profile == "cpu"
    assert created_configs[2].ctx_size == 4096
    assert created_configs[2].threads is None
    assert created_configs[2].threads_batch is None
    assert created_configs[2].no_mmproj_offload is None
    assert [entry.text for entry in written_entries] == ["AEC"]
    assert closed_roles == ["ocr", "ocr-fast", "corrector"]
    assert max_live_count == 1
    assert manifest[0]["corrector_prompt_style"] == "strict_ocr_v1"
    assert manifest[0]["conservative_merged_text"] == "AEC"
    assert manifest[0]["applied_op_count"] == 1


def test_convert_sup_to_srt_applies_qwen_mmproj_offload_override_when_requested(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    input_sup.write_bytes(b"")

    frames = [
        SimpleNamespace(
            raw_index=10,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=10),
            image=Image.new("RGB", (20, 2), "white"),
        )
    ]

    created_configs: list[OCRBackendConfig] = []

    class FakeBackend:
        def __init__(self, role: str) -> None:
            self.role = role

        def recognize_batch(self, images):
            if self.role == "ocr":
                return ["ABC"]
            if self.role == "ocr-fast":
                return ["ADC"]
            if self.role == "corrector":
                return ["AECZ"]
            raise AssertionError(f"unexpected role {self.role}")

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](len(frames))
        return iter(frames)

    def fake_create_backend(config: OCRBackendConfig):
        created_configs.append(config)
        return FakeBackend(config.role)

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", fake_create_backend)
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", lambda entries, path: path.write_text("", encoding="utf-8"))

    pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        engine=OCREngine.LLAMA_SERVER,
        corrector_config=CorrectorConfig(
            mode=CorrectorMode.QWEN_LOCAL,
            output_path=tmp_path / "corrected.jsonl",
            local_model_path=tmp_path / "qwen.gguf",
            local_mmproj_path=tmp_path / "qwen-mmproj.gguf",
            local_runtime_overrides=Qwen35RuntimeOverrides(no_mmproj_offload=True),
        ),
        srt_policy="overlap",
        verbose=False,
    )

    assert created_configs[2].role == "corrector"
    assert created_configs[2].no_mmproj_offload is True


def test_convert_sup_to_srt_applies_gemini_tall_prompt_gating(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    corrector_output = tmp_path / "gemini.jsonl"
    input_sup.write_bytes(b"")

    frames = [
        SimpleNamespace(
            raw_index=11,
            window_id=0,
            left=0,
            top=0,
            right=2,
            bottom=20,
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=10),
            image=Image.new("RGB", (2, 20), "white"),
        )
    ]

    created_roles: list[str] = []
    written_entries = []
    gemini_calls: list[str] = []

    class FakeBackend:
        def __init__(self, role: str) -> None:
            self.role = role

        def recognize_batch(self, images):
            if self.role == "ocr":
                return ["ABC"]
            if self.role == "detector":
                return ["ADC"]
            raise AssertionError(f"unexpected role {self.role}")

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](len(frames))
        return iter(frames)

    def fake_create_backend(config: OCRBackendConfig):
        created_roles.append(config.role)
        return FakeBackend(config.role)

    def fake_write_srt(entries, path):
        written_entries.extend(entries)
        path.write_text("", encoding="utf-8")

    def fake_request_gemini_correction(*, config, image, shape, verbose=False, abort_event=None):
        gemini_calls.append(shape)
        return "AECZ", "general_vertical_hint_v1", ""

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", fake_create_backend)
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", fake_write_srt)
    monkeypatch.setattr(pipeline, "request_gemini_correction", fake_request_gemini_correction)

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        engine=OCREngine.LLAMA_SERVER,
        corrector_config=CorrectorConfig(
            mode=CorrectorMode.GEMINI,
            output_path=corrector_output,
        ),
        srt_policy="overlap",
        verbose=False,
    )

    manifest = [json.loads(line) for line in corrector_output.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result.detector_record_count == 1
    assert result.correction_record_count == 1
    assert result.correction_applied_count == 1
    assert created_roles == ["ocr", "detector"]
    assert gemini_calls == ["tall"]
    assert [entry.text for entry in written_entries] == ["AEC"]
    assert manifest[0]["corrector_prompt_style"] == "general_vertical_hint_v1"


def test_convert_sup_to_srt_detector_reuses_exact_duplicate_images_within_branch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    detector_output = tmp_path / "detector.jsonl"
    input_sup.write_bytes(b"")

    frames = [
        SimpleNamespace(
            raw_index=10,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=10),
            image=Image.new("RGB", (20, 2), "white"),
        ),
        SimpleNamespace(
            raw_index=11,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=10),
            end=timedelta(milliseconds=20),
            image=Image.new("RGB", (20, 2), "white"),
        ),
        SimpleNamespace(
            raw_index=12,
            window_id=0,
            left=0,
            top=0,
            right=2,
            bottom=20,
            start=timedelta(milliseconds=20),
            end=timedelta(milliseconds=30),
            image=Image.new("RGB", (2, 20), "white"),
        ),
        SimpleNamespace(
            raw_index=13,
            window_id=0,
            left=0,
            top=0,
            right=2,
            bottom=20,
            start=timedelta(milliseconds=30),
            end=timedelta(milliseconds=40),
            image=Image.new("RGB", (2, 20), "white"),
        ),
    ]

    created_roles: list[str] = []
    role_calls: dict[str, int] = {}

    class FakeBackend:
        def __init__(self, role: str) -> None:
            self.role = role

        def recognize_batch(self, images):
            role_calls[self.role] = role_calls.get(self.role, 0) + 1
            if self.role == "ocr":
                return [["BASE-WIDE"], ["BASE-TALL"]][role_calls[self.role] - 1]
            if self.role == "ocr-fast":
                return ["ALT-WIDE"]
            if self.role == "detector":
                return ["ALT-TALL"]
            raise AssertionError(f"unexpected role {self.role}")

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](len(frames))
        return iter(frames)

    def fake_create_backend(config: OCRBackendConfig):
        created_roles.append(config.role)
        return FakeBackend(config.role)

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", fake_create_backend)
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", lambda entries, path: path.write_text("", encoding="utf-8"))

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        engine=OCREngine.LLAMA_SERVER,
        detector_output=detector_output,
        srt_policy="overlap",
        verbose=False,
    )

    manifest = [json.loads(line) for line in detector_output.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result.detector_record_count == 4
    assert created_roles == ["ocr", "ocr-fast", "detector"]
    assert role_calls == {"ocr": 2, "ocr-fast": 1, "detector": 1}
    assert [row["detector_branch"] for row in manifest] == [
        "alternate_read_non_tall",
        "alternate_read_non_tall",
        "repeat_drift_tall",
        "repeat_drift_tall",
    ]
    assert [row["baseline_text"] for row in manifest] == [
        "BASE-WIDE",
        "BASE-WIDE",
        "BASE-TALL",
        "BASE-TALL",
    ]
    assert [row["option_text"] for row in manifest] == [
        "ALT-WIDE",
        "ALT-WIDE",
        "ALT-TALL",
        "ALT-TALL",
    ]


def test_convert_sup_to_srt_qwen_corrector_reuses_exact_duplicate_images(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    corrector_output = tmp_path / "corrected.jsonl"
    input_sup.write_bytes(b"")

    frames = [
        SimpleNamespace(
            raw_index=10,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=10),
            image=Image.new("RGB", (20, 2), "white"),
        ),
        SimpleNamespace(
            raw_index=11,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=10),
            end=timedelta(milliseconds=20),
            image=Image.new("RGB", (20, 2), "white"),
        ),
    ]

    created_roles: list[str] = []
    role_calls: dict[str, int] = {}

    class FakeBackend:
        def __init__(self, role: str) -> None:
            self.role = role

        def recognize_batch(self, images):
            role_calls[self.role] = role_calls.get(self.role, 0) + 1
            if self.role == "ocr":
                return ["ABC"]
            if self.role == "ocr-fast":
                return ["ADC"]
            if self.role == "corrector":
                return ["AECZ"]
            raise AssertionError(f"unexpected role {self.role}")

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](len(frames))
        return iter(frames)

    def fake_create_backend(config: OCRBackendConfig):
        created_roles.append(config.role)
        return FakeBackend(config.role)

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", fake_create_backend)
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", lambda entries, path: path.write_text("", encoding="utf-8"))

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        engine=OCREngine.LLAMA_SERVER,
        corrector_config=CorrectorConfig(
            mode=CorrectorMode.QWEN_LOCAL,
            output_path=corrector_output,
            local_model_path=tmp_path / "qwen.gguf",
            local_mmproj_path=tmp_path / "qwen-mmproj.gguf",
            local_runtime_overrides=Qwen35RuntimeOverrides(profile="cpu"),
        ),
        srt_policy="overlap",
        verbose=False,
    )

    manifest = [json.loads(line) for line in corrector_output.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result.detector_record_count == 2
    assert result.correction_record_count == 2
    assert created_roles == ["ocr", "ocr-fast", "corrector"]
    assert role_calls == {"ocr": 1, "ocr-fast": 1, "corrector": 1}
    assert [row["corrector_text"] for row in manifest] == ["AECZ", "AECZ"]
    assert [row["conservative_merged_text"] for row in manifest] == ["AEC", "AEC"]


def test_recognize_prepared_inputs_reuses_exact_duplicate_images() -> None:
    white_a = Image.new("RGB", (2, 2), "white")
    white_b = Image.new("RGB", (2, 2), "white")
    black = Image.new("RGB", (2, 2), "black")
    prepared_inputs = [
        pipeline._PreparedOCRInput(index=0, frame=SimpleNamespace(raw_index=0), image=white_a),
        pipeline._PreparedOCRInput(index=1, frame=SimpleNamespace(raw_index=1), image=white_b),
        pipeline._PreparedOCRInput(index=2, frame=SimpleNamespace(raw_index=2), image=black),
    ]

    class FakeBackend:
        def __init__(self) -> None:
            self.calls = 0

        def recognize_batch(self, images):
            self.calls += 1
            return [f"call-{self.calls}"]

    backend = FakeBackend()

    texts = pipeline._recognize_prepared_inputs(  # noqa: SLF001
        prepared_inputs,
        backend=backend,
        verbose=False,
        branch_label="unit",
    )

    assert backend.calls == 2
    assert texts == ["call-1", "call-1", "call-2"]


def test_apply_gemini_corrections_reuses_exact_duplicate_images(monkeypatch) -> None:
    shared_a = Image.new("RGB", (20, 2), "white")
    shared_b = Image.new("RGB", (20, 2), "white")
    prepared_inputs = [
        pipeline._PreparedOCRInput(index=0, frame=SimpleNamespace(raw_index=100), image=shared_a),
        pipeline._PreparedOCRInput(index=1, frame=SimpleNamespace(raw_index=101), image=shared_b),
    ]
    detector_records = [
        HybridDetectorRecord(
            index=0,
            raw_index=100,
            window_id=0,
            start_ms=0,
            end_ms=10,
            detector_branch="alternate_read_non_tall",
            shape="wide",
            ratio=0.1,
            option_role="ocr-fast",
            baseline_text="ABC",
            option_text="ADC",
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=0.1,
        ),
        HybridDetectorRecord(
            index=1,
            raw_index=101,
            window_id=0,
            start_ms=10,
            end_ms=20,
            detector_branch="alternate_read_non_tall",
            shape="wide",
            ratio=0.1,
            option_role="ocr-fast",
            baseline_text="AEC",
            option_text="ADC",
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=0.1,
        ),
    ]

    gemini_calls: list[tuple[tuple[int, int], str]] = []

    def fake_request_gemini_correction(*, config, image, shape, verbose=False, abort_event=None):
        gemini_calls.append((image.size, shape))
        return "AXC", "strict_ocr_v1", ""

    monkeypatch.setattr(pipeline, "request_gemini_correction", fake_request_gemini_correction)

    records = pipeline._apply_gemini_corrections(  # noqa: SLF001
        prepared_inputs=prepared_inputs,
        detector_records=detector_records,
        corrector_config=CorrectorConfig(mode=CorrectorMode.GEMINI),
        verbose=False,
    )

    assert gemini_calls == [((20, 2), "wide")]
    assert len(records) == 2
    assert [record.corrector_text for record in records] == ["AXC", "AXC"]


def test_apply_gemini_corrections_does_not_reuse_when_shape_differs(monkeypatch) -> None:
    shared_a = Image.new("RGB", (20, 2), "white")
    shared_b = Image.new("RGB", (20, 2), "white")
    prepared_inputs = [
        pipeline._PreparedOCRInput(index=0, frame=SimpleNamespace(raw_index=100), image=shared_a),
        pipeline._PreparedOCRInput(index=1, frame=SimpleNamespace(raw_index=101), image=shared_b),
    ]
    detector_records = [
        HybridDetectorRecord(
            index=0,
            raw_index=100,
            window_id=0,
            start_ms=0,
            end_ms=10,
            detector_branch="alternate_read_non_tall",
            shape="wide",
            ratio=0.1,
            option_role="ocr-fast",
            baseline_text="ABC",
            option_text="ADC",
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=0.1,
        ),
        HybridDetectorRecord(
            index=1,
            raw_index=101,
            window_id=0,
            start_ms=10,
            end_ms=20,
            detector_branch="repeat_drift_tall",
            shape="tall",
            ratio=2.0,
            option_role="detector",
            baseline_text="AEC",
            option_text="ADC",
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=0.1,
        ),
    ]

    gemini_calls: list[tuple[tuple[int, int], str]] = []

    def fake_request_gemini_correction(*, config, image, shape, verbose=False, abort_event=None):
        gemini_calls.append((image.size, shape))
        return f"{shape}-TXT", "strict_ocr_v1", ""

    monkeypatch.setattr(pipeline, "request_gemini_correction", fake_request_gemini_correction)

    records = pipeline._apply_gemini_corrections(  # noqa: SLF001
        prepared_inputs=prepared_inputs,
        detector_records=detector_records,
        corrector_config=CorrectorConfig(mode=CorrectorMode.GEMINI),
        verbose=False,
    )

    assert gemini_calls == [((20, 2), "wide"), ((20, 2), "tall")]
    assert [record.corrector_text for record in records] == ["wide-TXT", "tall-TXT"]


def test_collect_gemini_correction_responses_runs_in_parallel_and_preserves_order(monkeypatch) -> None:
    prepared_by_index = {
        0: pipeline._PreparedOCRInput(
            index=0,
            frame=SimpleNamespace(raw_index=100),
            image=Image.new("RGB", (20, 2), "white"),
        ),
        1: pipeline._PreparedOCRInput(
            index=1,
            frame=SimpleNamespace(raw_index=101),
            image=Image.new("RGB", (20, 2), "black"),
        ),
    }
    unique_records = [
        HybridDetectorRecord(
            index=0,
            raw_index=100,
            window_id=0,
            start_ms=0,
            end_ms=10,
            detector_branch="alternate_read_non_tall",
            shape="wide",
            ratio=0.1,
            option_role="ocr-fast",
            baseline_text="ABC",
            option_text="ADC",
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=0.1,
        ),
        HybridDetectorRecord(
            index=1,
            raw_index=101,
            window_id=0,
            start_ms=10,
            end_ms=20,
            detector_branch="repeat_drift_tall",
            shape="tall",
            ratio=2.0,
            option_role="detector",
            baseline_text="AEC",
            option_text="ADC",
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=0.1,
        ),
    ]
    barrier = threading.Barrier(2, timeout=1.0)
    thread_names: set[str] = set()

    def fake_request_gemini_correction(*, config, image, shape, verbose=False, abort_event=None):
        thread_names.add(threading.current_thread().name)
        barrier.wait()
        return f"{shape}-TXT", "strict_ocr_v1", ""

    monkeypatch.setattr(pipeline, "request_gemini_correction", fake_request_gemini_correction)

    responses = pipeline._collect_gemini_correction_responses(  # noqa: SLF001
        unique_records=unique_records,
        prepared_by_index=prepared_by_index,
        corrector_config=CorrectorConfig(
            mode=CorrectorMode.GEMINI,
            gemini_max_workers=2,
            gemini_parallel_min_rows=1,
        ),
        verbose=False,
    )

    assert len(thread_names) == 2
    assert [response.corrector_text for response in responses] == ["wide-TXT", "tall-TXT"]
    assert [response.status for response in responses] == ["applied", "applied"]


def test_convert_sup_to_srt_keeps_baseline_for_failed_gemini_rows(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    corrector_output = tmp_path / "corrected.jsonl"
    input_sup.write_bytes(b"")

    frames = [
        SimpleNamespace(
            raw_index=10,
            window_id=0,
            left=0,
            top=0,
            right=20,
            bottom=2,
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=10),
            image=Image.new("RGB", (20, 2), "white"),
        ),
        SimpleNamespace(
            raw_index=11,
            window_id=0,
            left=0,
            top=0,
            right=2,
            bottom=20,
            start=timedelta(milliseconds=10),
            end=timedelta(milliseconds=20),
            image=Image.new("RGB", (2, 20), "white"),
        ),
    ]
    written_entries = []

    class FakeBackend:
        def __init__(self, role: str) -> None:
            self.role = role

        def recognize_batch(self, images):
            if self.role == "ocr":
                return ["ABC" if image.size == (20, 2) else "DEF" for image in images]
            if self.role == "ocr-fast":
                return ["AXC" if image.size == (20, 2) else "DZF" for image in images]
            if self.role == "detector":
                return ["AXC" if image.size == (20, 2) else "DZF" for image in images]
            raise AssertionError(f"unexpected role {self.role}")

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](len(frames))
        return iter(frames)

    def fake_create_backend(config: OCRBackendConfig):
        return FakeBackend(config.role)

    def fake_write_srt(entries, path):
        written_entries.extend(entries)
        path.write_text("", encoding="utf-8")

    def fake_request_gemini_correction(*, config, image, shape, verbose=False, abort_event=None):
        if shape == "wide":
            return "AYC", "strict_ocr_v1", "reasoning"
        raise GeminiRequestFailedError("http_503")

    monkeypatch.setattr(pipeline, "resolve_hf_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "create_ocr_backend", fake_create_backend)
    monkeypatch.setattr(pipeline, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(pipeline, "write_srt", fake_write_srt)
    monkeypatch.setattr(pipeline, "request_gemini_correction", fake_request_gemini_correction)

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        engine=OCREngine.LLAMA_SERVER,
        corrector_config=CorrectorConfig(
            mode=CorrectorMode.GEMINI,
            output_path=corrector_output,
            gemini_parallel_min_rows=1,
        ),
        srt_policy="overlap",
        verbose=False,
    )

    manifest = [json.loads(line) for line in corrector_output.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result.correction_record_count == 2
    assert result.correction_applied_count == 1
    assert result.correction_fallback_count == 1
    assert [entry.text for entry in written_entries] == ["AYC", "DEF"]
    assert [row["corrector_status"] for row in manifest] == ["applied", "fallback_baseline"]
    assert manifest[1]["corrector_error"] == "http_503"
    assert manifest[1]["conservative_merged_text"] == "DEF"


def test_apply_gemini_corrections_still_fails_fast_on_configuration_error(monkeypatch) -> None:
    prepared_inputs = [
        pipeline._PreparedOCRInput(
            index=0,
            frame=SimpleNamespace(raw_index=100),
            image=Image.new("RGB", (20, 2), "white"),
        )
    ]
    detector_records = [
        HybridDetectorRecord(
            index=0,
            raw_index=100,
            window_id=0,
            start_ms=0,
            end_ms=10,
            detector_branch="alternate_read_non_tall",
            shape="wide",
            ratio=0.1,
            option_role="ocr-fast",
            baseline_text="ABC",
            option_text="ADC",
            diff_label="meaningful_difference",
            meaningful=True,
            char_error_rate=0.1,
        )
    ]

    def fake_request_gemini_correction(*, config, image, shape, verbose=False, abort_event=None):
        raise GeminiConfigurationError("missing Gemini API key")

    monkeypatch.setattr(pipeline, "request_gemini_correction", fake_request_gemini_correction)

    with pytest.raises(GeminiConfigurationError, match="missing Gemini API key"):
        pipeline._apply_gemini_corrections(  # noqa: SLF001
            prepared_inputs=prepared_inputs,
            detector_records=detector_records,
            corrector_config=CorrectorConfig(mode=CorrectorMode.GEMINI),
            verbose=False,
        )


def test_merge_window_segments_splits_timeline_and_merges_active_texts() -> None:
    segments = [
        pipeline._WindowTextSegment(
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=20),
            text="VERT",
            window_id=1,
            left=100,
            top=0,
            right=110,
            bottom=50,
        ),
        pipeline._WindowTextSegment(
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=10),
            text="TOP",
            window_id=0,
            left=0,
            top=100,
            right=50,
            bottom=120,
        ),
        pipeline._WindowTextSegment(
            start=timedelta(milliseconds=10),
            end=timedelta(milliseconds=20),
            text="BOTTOM",
            window_id=0,
            left=0,
            top=100,
            right=50,
            bottom=120,
        ),
    ]

    entries = pipeline._merge_window_segments(segments)  # noqa: SLF001

    assert [(entry.start, entry.end, entry.text) for entry in entries] == [
        (timedelta(milliseconds=0), timedelta(milliseconds=10), "VERT\nTOP"),
        (timedelta(milliseconds=10), timedelta(milliseconds=20), "VERT\nBOTTOM"),
    ]


def test_overlap_window_segments_keeps_overlapping_cues_separate() -> None:
    segments = [
        pipeline._WindowTextSegment(
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=20),
            text="VERT",
            window_id=1,
            left=100,
            top=0,
            right=110,
            bottom=50,
        ),
        pipeline._WindowTextSegment(
            start=timedelta(milliseconds=0),
            end=timedelta(milliseconds=10),
            text="TOP",
            window_id=0,
            left=0,
            top=100,
            right=50,
            bottom=120,
        ),
        pipeline._WindowTextSegment(
            start=timedelta(milliseconds=10),
            end=timedelta(milliseconds=20),
            text="BOTTOM",
            window_id=0,
            left=0,
            top=100,
            right=50,
            bottom=120,
        ),
    ]

    entries = pipeline._build_subtitle_entries(segments, srt_policy="overlap")  # noqa: SLF001

    assert [(entry.start, entry.end, entry.text) for entry in entries] == [
        (timedelta(milliseconds=0), timedelta(milliseconds=20), "VERT"),
        (timedelta(milliseconds=0), timedelta(milliseconds=10), "TOP"),
        (timedelta(milliseconds=10), timedelta(milliseconds=20), "BOTTOM"),
    ]


def test_build_subtitle_entries_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="unsupported srt policy"):
        pipeline._build_subtitle_entries([], srt_policy="bad-mode")  # noqa: SLF001
