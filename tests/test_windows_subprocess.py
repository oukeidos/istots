from __future__ import annotations

from istots import windows_subprocess


def test_hidden_windows_subprocess_kwargs_include_no_window_flags_on_windows() -> None:
    kwargs = windows_subprocess.hidden_windows_subprocess_kwargs(new_process_group=True)

    assert kwargs["creationflags"] & getattr(windows_subprocess.subprocess, "CREATE_NO_WINDOW", 0)
    assert kwargs["creationflags"] & getattr(
        windows_subprocess.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        0,
    )
    assert kwargs["startupinfo"].dwFlags & getattr(
        windows_subprocess.subprocess,
        "STARTF_USESHOWWINDOW",
        0,
    )


def test_hidden_windows_subprocess_kwargs_omit_new_process_group_when_not_requested() -> None:
    kwargs = windows_subprocess.hidden_windows_subprocess_kwargs()

    assert kwargs["creationflags"] & getattr(windows_subprocess.subprocess, "CREATE_NO_WINDOW", 0)
    assert not (
        kwargs["creationflags"]
        & getattr(windows_subprocess.subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    )
