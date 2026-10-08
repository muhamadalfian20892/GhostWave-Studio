; Inno Setup Script for GhostWave Studio v1.0
; Build standalone Windows installer package

#define MyAppName "GhostWave Studio"
#define MyAppVersion "1.0"
#define MyAppPublisher "Technokers Lab"
#define MyAppURL "https://github.com/muhamadalfian20892/GhostWave-Studio"
#define MyAppExeName "GhostWaveStudio.exe"
#define MyAppCliName "ghostwave-cli.exe"

[Setup]
AppId={{B8F36990-2C5E-4B07-9E45-81A2BCF7D891}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=LICENSE
OutputDir=dist\installer
OutputBaseFilename=GhostWaveStudio-v1.0-Setup
SetupIconFile=ghostwave.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\{#MyAppExeName}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "addtopath"; Description: "Add GhostWave CLI to User PATH (allows running 'ghostwave-cli' from terminal)"; GroupDescription: "Command Line Integration:"; Flags: unchecked

[Files]
Source: "dist\GhostWaveStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "CONTRIBUTING.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "ghostwave.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\ghostwave.ico"
Name: "{group}\GhostWave CLI"; Filename: "{app}\{#MyAppCliName}"; IconFilename: "{app}\ghostwave.ico"
Name: "{group}\Documentation"; Filename: "{app}\README.md"
Name: "{group}\Contributing Guidelines"; Filename: "{app}\CONTRIBUTING.md"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; IconFilename: "{app}\ghostwave.ico"

[Registry]
Root: HKCU; Subkey: "Environment"; \
    ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; \
    Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Code]
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + UpperCase(Param) + ';', ';' + UpperCase(OrigPath) + ';') = 0;
end;

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
