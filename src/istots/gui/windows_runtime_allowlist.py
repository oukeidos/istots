from __future__ import annotations

MANUAL_MANAGED_RUNTIME_CANDIDATE_LIMIT = 3
AUTO_MANAGED_RUNTIME_CANDIDATE_LIMIT = 4
AUTO_MANAGED_RUNTIME_FAMILY_ORDER = (
    "x64/cuda12",
    "x64/vulkan",
    "x64/cpu",
)

WINDOWS_RUNTIME_ALLOWLIST_BY_VARIANT: dict[str, tuple[str, ...]] = {
    "x64/cpu": (
        "b8887",
        "b8886",
        "b8885",
        "b8833",
        "b8832",
        "b8816",
        "b8813",
    ),
    "x64/cuda12": (
        "b8892",
        "b8885",
        "b8860",
        "b8833",
    ),
    "x64/vulkan": (
        "b8892",
        "b8891",
        "b8890",
        "b8889",
        "b8888",
        "b8887",
        "b8886",
        "b8885",
        "b8882",
        "b8871",
        "b8870",
        "b8854",
        "b8853",
        "b8852",
        "b8851",
        "b8846",
        "b8842",
        "b8841",
        "b8838",
        "b8837",
        "b8836",
        "b8833",
        "b8832",
        "b8831",
        "b8829",
        "b8828",
        "b8827",
        "b8826",
        "b8824",
        "b8822",
        "b8821",
        "b8816",
        "b8815",
        "b8813",
        "b8811",
        "b8809",
    ),
}

MANUAL_MANAGED_RUNTIME_VARIANTS = (
    "x64/cpu",
    "x64/cuda12",
    "x64/vulkan",
)


def allowlisted_runtime_tags_for_variant(variant_id: str) -> tuple[str, ...]:
    return WINDOWS_RUNTIME_ALLOWLIST_BY_VARIANT.get(variant_id, ())
