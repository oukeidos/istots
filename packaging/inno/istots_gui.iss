#define protected MyAppName "IStoTS"
#define protected MyAppPublisher "oukeidos"
#define protected MyAppId "{{07ac00d9-1e18-4ee9-8af6-01c007408576}"

#ifndef MyAppVersion
  #error "MyAppVersion must be defined by the build script."
#endif

#ifndef MyBundleRoot
  #error "MyBundleRoot must be defined by the build script."
#endif

#ifndef MyOutputDir
  #define protected MyOutputDir "Output"
#endif

#ifndef MyOutputBaseFilename
  #define protected MyOutputBaseFilename MyAppName + "-" + MyAppVersion + "-windows-x64-setup"
#endif

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://github.com/oukeidos/istots
AppSupportURL=https://github.com/oukeidos/istots
AppUpdatesURL=https://github.com/oukeidos/istots/releases
DefaultDirName={autopf}\{#MyAppName}
UsePreviousAppDir=yes
UsePreviousTasks=yes
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\..\src\istots\resources\icons\windows\istots_setup.ico
UninstallDisplayIcon={app}\istots.ico
LicenseFile=..\..\LICENSE
OutputDir={#MyOutputDir}
OutputBaseFilename={#MyOutputBaseFilename}

[Tasks]
; Install the desktop shortcut by default on first install, then preserve the user's choice.
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; Keep managed runtime and model assets outside {app}; install only the built GUI bundle.
Source: "{#MyBundleRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\istots.exe"; IconFilename: "{app}\istots.ico"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\istots.exe"; IconFilename: "{app}\istots.ico"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\istots.exe"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent unchecked

[Code]
var
  ShouldRemoveManagedAssets: Boolean;

function ManagedAssetsDir(): String;
begin
  Result := ExpandConstant('{localappdata}\istots\managed');
end;

function InitializeUninstall(): Boolean;
begin
  Result := True;
  ShouldRemoveManagedAssets := False;

  if UninstallSilent() then
    exit;

  if not DirExists(ManagedAssetsDir()) then
    exit;

  if MsgBox(
    'Remove downloaded managed runtime and model assets under:' + #13#10 + #13#10 +
    ManagedAssetsDir() + #13#10 + #13#10 +
    'Choose No to keep them for a later reinstall or upgrade.',
    mbConfirmation,
    MB_YESNO
  ) = IDYES then
    ShouldRemoveManagedAssets := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep <> usPostUninstall then
    exit;
  if not ShouldRemoveManagedAssets then
    exit;
  if not DirExists(ManagedAssetsDir()) then
    exit;

  if not DelTree(ManagedAssetsDir(), True, True, True) then
    MsgBox(
      'IStoTS could not remove every managed asset under:' + #13#10 + #13#10 +
      ManagedAssetsDir(),
      mbError,
      MB_OK
    );
end;
