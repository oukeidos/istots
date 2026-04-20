[Setup]
; Use at compile time
SetupIconFile=packaging/icons/windows/istots_setup.ico
; Use after install for Add/Remove Programs
UninstallDisplayIcon={app}\istots.ico

[Icons]
; Start Menu shortcut
Name: "{group}\istots"; Filename: "{app}\istots.exe"; IconFilename: "{app}\istots.ico"
; Optional desktop shortcut example
Name: "{commondesktop}\istots"; Filename: "{app}\istots.exe"; IconFilename: "{app}\istots.ico"; Tasks: desktopicon
