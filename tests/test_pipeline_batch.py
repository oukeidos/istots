from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from istots import pipeline
from istots.corrector import CorrectorConfig, CorrectorMode
from istots.ocr import OCRBackendConfig, OCREngine


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

    with pytest.raises(ValueError, match="fast OCR mode requires the llama-server engine"):
        pipeline.convert_sup_to_srt(
            input_sup=input_sup,
            output_srt=output_srt,
            engine=OCREngine.HF,
            ocr_mode="fast",
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
    assert [row["diff_label"] for row in manifest] == [
        "meaningful_difference",
        "meaningful_difference",
    ]


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
        runtime_profile="cpu",
        corrector_config=CorrectorConfig(
            mode=CorrectorMode.QWEN_LOCAL,
            output_path=corrector_output,
            local_model_path=tmp_path / "qwen.gguf",
            local_mmproj_path=tmp_path / "qwen-mmproj.gguf",
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
    assert created_configs[2].no_mmproj_offload is False
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
            local_no_mmproj_offload=True,
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

    def fake_request_gemini_correction(*, config, image, shape):
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
