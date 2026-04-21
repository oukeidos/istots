[Setup]
; Use at compile time
SetupIconFile=src\istots\resources\icons\windows\istots_setup.ico
; Use after install for Add/Remove Programs
UninstallDisplayIcon={app}\istots.ico

[Icons]
; Start Menu shortcut
Name: "{group}\IStoTS"; Filename: "{app}\istots.exe"; IconFilename: "{app}\istots.ico"; WorkingDir: "{app}"
; Optional desktop shortcut example
Name: "{autodesktop}\IStoTS"; Filename: "{app}\istots.exe"; IconFilename: "{app}\istots.ico"; WorkingDir: "{app}"; Tasks: desktopicon
