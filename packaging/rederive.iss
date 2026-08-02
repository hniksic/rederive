; The Windows installer: the directory form of the program, in a wizard.
;
; Built on Windows, from the repository root, after the tree build has run:
;
;     iscc /DAppVersion=0.1.0 packaging\rederive.iss
;
; which leaves `dist\rederive-setup.exe`. `RELEASING.md` says where that goes.
;
; The install is per-user by default and asks for no elevation; a user who wants it
; for the whole machine can say so in the wizard's first page. Rederive is a terminal
; program, so the point of installing it is the PATH entry rather than the Start Menu
; shortcut, and both are removed again on uninstall.

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#define AppName "Rederive"
#define AppExe "rederive.exe"

[Setup]
AppId={{946A6FD4-8329-44D6-B82C-C798374F82C6}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Hrvoje Niksic
AppPublisherURL=https://github.com/hniksic/rederive
AppSupportURL=https://github.com/hniksic/rederive/issues
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=rederive-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ChangesEnvironment=yes
UninstallDisplayIcon={app}\{#AppExe}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath"; Description: "Add Rederive to PATH, so that `rederive` starts it"

[Files]
Source: "..\dist\tree\rederive\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"

[Registry]
Root: HKA; Subkey: "{code:EnvironmentKey}"; ValueType: expandsz; ValueName: "Path"; \
    ValueData: "{olddata};{app}"; Flags: preservestringtype; Tasks: addtopath; \
    Check: NeedsAddPath(ExpandConstant('{app}'))

[Run]
Filename: "{app}\{#AppExe}"; Description: "Start Rederive"; \
    Flags: postinstall nowait skipifsilent unchecked

[Code]

{ Where PATH lives depends on who the install is for, so the key cannot be a
  constant: a per-user install writes the user's environment, a machine-wide one
  writes the system's. }
function EnvironmentKey(Param: string): string;
begin
  if IsAdminInstallMode then
    Result := 'System\CurrentControlSet\Control\Session Manager\Environment'
  else
    Result := 'Environment';
end;

function NeedsAddPath(Directory: string): boolean;
var
  Existing: string;
begin
  if not RegQueryStringValue(HKA, EnvironmentKey(''), 'Path', Existing) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Directory) + ';', ';' + Uppercase(Existing) + ';') = 0;
end;

{ Take the directory back out of PATH on the way out, leaving the rest of it as it
  was found. The search is done on a string padded with separators at both ends, so
  that an entry at either end is matched the same way as one in the middle. }
procedure RemoveFromPath(Directory: string);
var
  Existing: string;
  Position: Integer;
begin
  if not RegQueryStringValue(HKA, EnvironmentKey(''), 'Path', Existing) then
    exit;
  Position := Pos(';' + Uppercase(Directory) + ';', ';' + Uppercase(Existing) + ';');
  if Position = 0 then
    exit;
  if Position = 1 then
    Delete(Existing, 1, Length(Directory) + 1)
  else
    Delete(Existing, Position - 1, Length(Directory) + 1);
  RegWriteExpandStringValue(HKA, EnvironmentKey(''), 'Path', Existing);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveFromPath(ExpandConstant('{app}'));
end;
