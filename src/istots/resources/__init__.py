"""Packaged non-code resources for istots."""

from importlib.resources import abc, files

_GUI_ICON_PNG_NAMES = (
    "istots_16.png",
    "istots_32.png",
    "istots_48.png",
    "istots_256.png",
    "istots_512.png",
)


def icon_bundle_root() -> abc.Traversable:
    return files(__name__).joinpath("icons")


def iter_gui_icon_png_payloads() -> tuple[bytes, ...]:
    base = icon_bundle_root().joinpath("png", "generic")
    return tuple(base.joinpath(name).read_bytes() for name in _GUI_ICON_PNG_NAMES)
