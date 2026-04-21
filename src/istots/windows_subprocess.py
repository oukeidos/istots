from __future__ import annotations

import os
import subprocess
from typing import Any


def hidden_windows_subprocess_kwargs(*, new_process_group: bool = False) -> dict[str, Any]:
    if os.name != "nt":
        return {}

    kwargs: dict[str, Any] = {}
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if new_process_group:
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    if creationflags:
        kwargs["creationflags"] = creationflags

    startupinfo_type = getattr(subprocess, "STARTUPINFO", None)
    startf_use_showwindow = getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
    if startupinfo_type is not None and startf_use_showwindow:
        startupinfo = startupinfo_type()
        startupinfo.dwFlags |= startf_use_showwindow
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        kwargs["startupinfo"] = startupinfo

    return kwargs
