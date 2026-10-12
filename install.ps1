# ---------------------------------------------------------------------------
# Suzy one-click installer for Windows 10/11
#
# Run this in PowerShell (no admin needed):
#
#   powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/sagar88776h/Suzi-my-personal-ai-assistant/refs/heads/claude/youthful-sagan-m97rqy/install.ps1 | iex"
#
# What it does:
#   1. Finds Python 3.11-3.13, or installs Python 3.12 for the current user.
#   2. Downloads Suzy into %USERPROFILE%\Suzy (re-running it updates the code
#      but keeps your .env keys and config.yaml).
#   3. Creates a virtual environment and installs the requirements.
#   4. Asks for your free Groq / Gemini API keys and saves them to .env.
#   5. Runs the test suite, creates a "Suzy" desktop shortcut, and starts Suzy.
# ---------------------------------------------------------------------------

$Repo       = 'sagar88776h/Suzi-my-personal-ai-assistant'
$Branch     = if ($env:SUZY_BRANCH) { $env:SUZY_BRANCH } else { 'claude/youthful-sagan-m97rqy' }
$InstallDir = Join-Path $HOME 'Suzy'
$PythonInstallerUrl = 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe'

$ErrorActionPreference = 'Continue'   # native tools write warnings to stderr; we check exit codes instead
$ProgressPreference    = 'SilentlyContinue'  # makes downloads much faster in Windows PowerShell
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

function Write-Step($text)  { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Write-Ok($text)    { Write-Host "    $text" -ForegroundColor Green }
function Write-Note($text)  { Write-Host "    $text" -ForegroundColor Yellow }

function Confirm-No($question) {
    # Returns $true only if the user types y / yes.
    $answer = Read-Host "    $question (y/N)"
    return ($answer -match '^\s*(y|yes|haan|ha)\s*$')
}

function Confirm-Default($question) {
    # Returns $true unless the user types n / no.
    $answer = Read-Host "    $question (Y/n)"
    return -not ($answer -match '^\s*(n|no)\s*$')
}

# --------------------------------------------------------------------------- Python
function Test-PythonCandidate($exe, [string[]]$launcherArgs) {
    # Returns the full path of a usable Python (3.11 - 3.13), or $null.
    try {
        $out = & $exe @launcherArgs -c "import sys; print('%d.%d' % sys.version_info[:2]); print(sys.executable)" 2>$null
    } catch {
        return $null
    }
    if ($LASTEXITCODE -ne 0 -or -not $out -or $out.Count -lt 2) { return $null }
    try { $version = [version]$out[0] } catch { return $null }
    if ($version -ge [version]'3.11' -and $version -lt [version]'3.14') { return $out[1].Trim() }
    return $null
}

function Find-Python {
    $candidates = @(
        @{ Exe = 'py'; Args = @('-3.12') },
        @{ Exe = 'py'; Args = @('-3.13') },
        @{ Exe = 'py'; Args = @('-3.11') },
        @{ Exe = 'python'; Args = @() },
        @{ Exe = 'python3'; Args = @() }
    )
    foreach ($minor in '312', '313', '311') {
        $candidates += @{ Exe = (Join-Path $env:LOCALAPPDATA "Programs\Python\Python$minor\python.exe"); Args = @() }
        $candidates += @{ Exe = (Join-Path $env:ProgramFiles "Python$minor\python.exe"); Args = @() }
    }
    foreach ($candidate in $candidates) {
        $isPath = $candidate.Exe -like '*\*'
        if ($isPath -and -not (Test-Path $candidate.Exe)) { continue }
        if (-not $isPath -and -not (Get-Command $candidate.Exe -ErrorAction SilentlyContinue)) { continue }
        $found = Test-PythonCandidate $candidate.Exe $candidate.Args
        if ($found) { return $found }
    }
    return $null
}

function Install-Python {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Note "Installing Python 3.12 with winget (this can take a few minutes)..."
        & winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
        $found = Find-Python
        if ($found) { return $found }
        Write-Note "winget did not finish the install; downloading Python from python.org instead..."
    }
    $installer = Join-Path $env:TEMP 'python-3.12-installer.exe'
    Invoke-WebRequest -Uri $PythonInstallerUrl -OutFile $installer -UseBasicParsing -ErrorAction Stop
    Write-Note "Running the Python installer quietly (just for your user, no admin needed)..."
    $proc = Start-Process -FilePath $installer -Wait -PassThru -ArgumentList @(
        '/quiet', 'InstallAllUsers=0', 'PrependPath=1', 'Include_launcher=0', 'Include_test=0'
    )
    Remove-Item $installer -Force -ErrorAction SilentlyContinue
    if ($proc.ExitCode -ne 0) { throw "The Python installer failed with exit code $($proc.ExitCode)." }
    return Find-Python
}

# --------------------------------------------------------------------------- .env helpers
function Read-EnvFile($path) {
    $values = @{}
    if (Test-Path $path) {
        foreach ($line in Get-Content $path -ErrorAction SilentlyContinue) {
            if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
                $values[$matches[1]] = $matches[2].Trim().Trim('"').Trim("'")
            }
        }
    }
    return $values
}

function Write-EnvFile($path, $values) {
    # Known keys with their comments, then every other entry already in the file (never lost).
    $known = @(
        @('GROQ_API_KEY', '# Groq (main brain + fast speech recognition): https://console.groq.com/keys'),
        @('GEMINI_API_KEY', '# Google Gemini (backup brain): https://aistudio.google.com/apikey'),
        @('OLLAMA_API_KEY', '# Ollama Cloud (optional brain): https://ollama.com/settings/keys')
    )
    $lines = @('# Suzy API keys - keep this file private.')
    foreach ($entry in $known) {
        $lines += ''
        $lines += $entry[1]
        $lines += "$($entry[0])=$($values[$entry[0]])"
    }
    $knownNames = $known | ForEach-Object { $_[0] }
    $others = @($values.Keys | Where-Object { $knownNames -notcontains $_ } | Sort-Object)
    if ($others.Count -gt 0) {
        $lines += ''
        $lines += '# Other settings'
        foreach ($name in $others) { $lines += "$name=$($values[$name])" }
    }
    $lines += ''
    $text = $lines -join "`r`n"
    # UTF-8 *without* BOM: a BOM would corrupt the first key name for python-dotenv.
    [IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))
}

function Request-ApiKey($label, $url, $prefix) {
    Write-Host ""
    Write-Host "    $label - free key at: $url" -ForegroundColor White
    if (Confirm-Default "Open that page in your browser now?") { Start-Process $url }
    # $prefix may list several accepted prefixes separated by " or ".
    $hint = if ($prefix) { " (starts with $prefix)" } else { "" }
    $key = (Read-Host "    Paste your $label key here$hint, or just press Enter to skip").Trim().Trim('"').Trim("'")
    $known = @($prefix -split ' or ' | Where-Object { $key.StartsWith($_) })
    if ($key -and $prefix -and $known.Count -eq 0) {
        Write-Note "That doesn't start with $prefix - saving it anyway, double-check it if Suzy complains."
    }
    return $key
}

# --------------------------------------------------------------------------- main
function Install-Suzy {
    Write-Host ""
    Write-Host "  Suzy installer" -ForegroundColor Magenta
    Write-Host "  Installing into $InstallDir" -ForegroundColor Magenta

    # 1. Python ------------------------------------------------------------
    Write-Step "Step 1/6: Checking Python"
    $python = Find-Python
    if (-not $python) {
        Write-Note "No suitable Python (3.11 - 3.13) found."
        $python = Install-Python
        if (-not $python) {
            throw "Python was installed but could not be found. Close this window, open a new PowerShell and run the installer again."
        }
    }
    Write-Ok "Using Python: $python"

    # 2. Code --------------------------------------------------------------
    Write-Step "Step 2/6: Downloading Suzy"
    $zip = Join-Path $env:TEMP 'suzy-source.zip'
    $extract = Join-Path $env:TEMP 'suzy-source'
    Remove-Item $extract -Recurse -Force -ErrorAction SilentlyContinue
    Invoke-WebRequest -Uri "https://github.com/$Repo/archive/refs/heads/$Branch.zip" -OutFile $zip -UseBasicParsing -ErrorAction Stop
    Expand-Archive -Path $zip -DestinationPath $extract -Force -ErrorAction Stop
    $top = Get-ChildItem $extract -Directory | Select-Object -First 1
    $source = Join-Path $top.FullName 'suzy'
    if (-not (Test-Path (Join-Path $source 'main.py'))) { throw "The download did not contain Suzy's code." }

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    # Suzy may have changed her own code. This update replaces files it ships, so keep a
    # full copy of her current code first. New abilities she added are never deleted.
    if (Test-Path (Join-Path $InstallDir 'data\self_changes.jsonl')) {
        $selfBackup = Join-Path $InstallDir ("backups\installer-update-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
        & robocopy $InstallDir $selfBackup /E /XD .venv data logs models backups __pycache__ /XF .env /NFL /NDL /NJH /NJS /NP /R:1 /W:1 | Out-Null
        Write-Note "Suzy had changed her own code - saved a copy in $selfBackup before updating."
    }
    # Copy everything except your private/local files (.env, config.yaml, venv, logs, data, backups).
    & robocopy $source $InstallDir /E /XD .venv data logs models backups __pycache__ /XF .env config.yaml /NFL /NDL /NJH /NJS /NP /R:1 /W:1 | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copying Suzy's files failed (robocopy code $LASTEXITCODE)." }
    $configPath = Join-Path $InstallDir 'config.yaml'
    $newConfig = Join-Path $source 'config.yaml'
    if (-not (Test-Path $configPath)) {
        Copy-Item $newConfig $configPath
    } elseif (Select-String -Path $configPath -Pattern 'llama-3\.3-70b-versatile|^\s+model:\s*gemini-2\.5-flash' -Quiet) {
        # Settings from the first release point at AI models that have since been retired.
        # Swap in the new config.yaml, but keep the name Suzy calls you by.
        $backup = Join-Path $InstallDir 'config.backup.yaml'
        Copy-Item $configPath $backup -Force
        $oldName = $null
        $nameLine = Select-String -Path $configPath -Pattern '^\s+name:\s*([^#]+?)\s*(#.*)?$' | Select-Object -First 1
        if ($nameLine) { $oldName = $nameLine.Matches[0].Groups[1].Value.Trim() }
        $text = [IO.File]::ReadAllText($newConfig)
        if ($oldName) {
            $text = ([regex]'(?m)^(\s+name:[ \t]*)[^#\r\n]+?([ \t]*#)').Replace($text, "`${1}$oldName`${2}", 1)
        }
        [IO.File]::WriteAllText($configPath, $text, (New-Object Text.UTF8Encoding $false))
        Write-Ok "Updated config.yaml to the current AI models (old one saved as config.backup.yaml)."
    } else {
        Write-Ok "Kept your existing config.yaml."
        # Move settings that still hold an OLD default to the new default (values you chose are kept).
        $configText = [IO.File]::ReadAllText($configPath)
        $migrated = $configText
        $migrated = [regex]::Replace($migrated, '(?m)^(\s+voice:[ \t]*)en-IN-NeerjaNeural([ \t]+# also try[^\r\n]*)(?=\r?$)',
            '${1}en-IN-NeerjaExpressiveNeural  # hear all voices: double-click "Suzy Voices"')
        $migrated = [regex]::Replace($migrated, '(?m)^(\s+fallback_voice:[ \t]*)en-US-AriaNeural[ \t]*(?=\r?$)', '${1}en-IN-NeerjaNeural')
        if ($migrated -ne $configText) {
            [IO.File]::WriteAllText($configPath, $migrated, (New-Object Text.UTF8Encoding $false))
            Write-Ok "Switched Suzy to the new, more natural Indian voice."
        }
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir 'logs'), (Join-Path $InstallDir 'data') | Out-Null
    Remove-Item $zip, $extract -Recurse -Force -ErrorAction SilentlyContinue
    Write-Ok "Suzy's code is in $InstallDir"

    # 3. Virtual environment + requirements --------------------------------
    Write-Step "Step 3/6: Installing Suzy's libraries (a few minutes the first time)"
    $venvDir = Join-Path $InstallDir '.venv'
    $venvPython = Join-Path $venvDir 'Scripts\python.exe'
    if (Test-Path $venvPython) {
        & $venvPython -c "import sys" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Note "The existing virtual environment is broken; recreating it."
            Remove-Item $venvDir -Recurse -Force
        }
    }
    if (-not (Test-Path $venvPython)) {
        & $python -m venv $venvDir
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) { throw "Could not create the virtual environment." }
    }
    & $venvPython -m pip install --upgrade pip --disable-pip-version-check --quiet
    & $venvPython -m pip install -r (Join-Path $InstallDir 'requirements.txt') --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { throw "Installing the requirements failed - see the messages above." }
    Write-Ok "Libraries installed."

    # 4. API keys ----------------------------------------------------------
    Write-Step "Step 4/6: API keys (both are free, no credit card)"
    $envPath = Join-Path $InstallDir '.env'
    $keys = Read-EnvFile $envPath
    if ($keys['GROQ_API_KEY'] -and $keys['GEMINI_API_KEY']) {
        Write-Ok "Found your saved keys."
        if (-not (Confirm-Default "Keep them?")) { $keys['GROQ_API_KEY'] = ''; $keys['GEMINI_API_KEY'] = '' }
    }
    if (-not $keys['GROQ_API_KEY']) {
        $keys['GROQ_API_KEY'] = Request-ApiKey 'Groq (main brain + fast listening)' 'https://console.groq.com/keys' 'gsk_'
    }
    if (-not $keys['GEMINI_API_KEY']) {
        $keys['GEMINI_API_KEY'] = Request-ApiKey 'Google Gemini (backup brain)' 'https://aistudio.google.com/apikey' 'AIza or AQ.'
    }
    if ($keys['OLLAMA_API_KEY']) {
        Write-Ok "Found your Ollama Cloud key too."
    } elseif (Confirm-No "Do you also have an Ollama Cloud API key to add (optional)?") {
        $keys['OLLAMA_API_KEY'] = Request-ApiKey 'Ollama Cloud (extra brain)' 'https://ollama.com/settings/keys' ''
    }
    $groqKey = $keys['GROQ_API_KEY']
    $geminiKey = $keys['GEMINI_API_KEY']
    Write-EnvFile $envPath $keys
    if (-not $groqKey -and -not $geminiKey -and -not $keys['OLLAMA_API_KEY']) {
        Write-Note "No keys saved - Suzy can't think without one. Re-run this installer or edit $envPath later."
    } else {
        Write-Ok "Keys saved to $envPath"
    }

    # 5. Tests -------------------------------------------------------------
    Write-Step "Step 5/6: Running Suzy's self-tests"
    Push-Location $InstallDir
    & $venvPython -m pytest -q -p no:cacheprovider
    $testsOk = ($LASTEXITCODE -eq 0)
    Pop-Location
    if ($testsOk) { Write-Ok "All tests passed." } else { Write-Note "Some tests failed - please send the output above to your developer." }

    # 6. Launcher + desktop shortcut ---------------------------------------
    Write-Step "Step 6/6: Creating the Suzy shortcut"
    $launcher = Join-Path $InstallDir 'Start Suzy.bat'
    $bat = @(
        '@echo off',
        'title Suzy',
        'cd /d "%~dp0"',
        'rem guard.py starts Suzy, restarts her after self-updates and rolls back broken ones.',
        '".venv\Scripts\python.exe" guard.py %*',
        'echo.',
        'pause'
    ) -join "`r`n"
    [IO.File]::WriteAllText($launcher, $bat, (New-Object Text.ASCIIEncoding))
    $micLauncher = Join-Path $InstallDir 'Suzy Mic Test.bat'
    $micBat = @(
        '@echo off',
        'title Suzy microphone test',
        'cd /d "%~dp0"',
        '".venv\Scripts\python.exe" main.py --mic-test',
        'echo.',
        'pause'
    ) -join "`r`n"
    [IO.File]::WriteAllText($micLauncher, $micBat, (New-Object Text.ASCIIEncoding))
    $voicesLauncher = Join-Path $InstallDir 'Suzy Voices.bat'
    $voicesBat = @(
        '@echo off',
        'title Suzy voices',
        'cd /d "%~dp0"',
        '".venv\Scripts\python.exe" main.py --voices',
        'echo.',
        'pause'
    ) -join "`r`n"
    [IO.File]::WriteAllText($voicesLauncher, $voicesBat, (New-Object Text.ASCIIEncoding))
    try {
        $desktop = [Environment]::GetFolderPath('Desktop')
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut((Join-Path $desktop 'Suzy.lnk'))
        $shortcut.TargetPath = $launcher
        $shortcut.WorkingDirectory = $InstallDir
        $shortcut.IconLocation = "$venvPython,0"
        $shortcut.Description = 'Start Suzy, your AI assistant'
        $shortcut.Save()
        $micShortcut = $shell.CreateShortcut((Join-Path $desktop 'Suzy Mic Test.lnk'))
        $micShortcut.TargetPath = $micLauncher
        $micShortcut.WorkingDirectory = $InstallDir
        $micShortcut.IconLocation = "$env:SystemRoot\System32\SndVol.exe,0"
        $micShortcut.Description = 'Check that Suzy can hear your microphone'
        $micShortcut.Save()
        $voicesShortcut = $shell.CreateShortcut((Join-Path $desktop 'Suzy Voices.lnk'))
        $voicesShortcut.TargetPath = $voicesLauncher
        $voicesShortcut.WorkingDirectory = $InstallDir
        $voicesShortcut.IconLocation = "$env:SystemRoot\System32\SndVol.exe,0"
        $voicesShortcut.Description = "Listen to Suzy's voices and pick your favourite"
        $voicesShortcut.Save()
        Write-Ok "Desktop shortcuts 'Suzy', 'Suzy Mic Test' and 'Suzy Voices' created."
    } catch {
        Write-Note "Couldn't create the desktop shortcut; you can double-click '$launcher' instead."
    }

    # Start with Windows: a minimised shortcut in the Startup folder, run with --startup so
    # Suzy greets "Hello <name>, welcome back."
    $startupLink = Join-Path ([Environment]::GetFolderPath('Startup')) 'Suzy.lnk'
    Write-Host ""
    if (Confirm-Default "Start Suzy automatically when Windows starts (she'll say 'welcome back')?") {
        try {
            $autoShell = New-Object -ComObject WScript.Shell
            $auto = $autoShell.CreateShortcut($startupLink)
            $auto.TargetPath = $launcher
            $auto.Arguments = '--startup'
            $auto.WorkingDirectory = $InstallDir
            $auto.WindowStyle = 7
            $auto.Description = 'Start Suzy when Windows starts'
            $auto.Save()
            Write-Ok "Suzy will start with Windows and welcome you back."
        } catch {
            Write-Note "Couldn't add Suzy to Windows startup. You can ask her: 'Suzy, start with Windows'."
        }
    } else {
        Remove-Item $startupLink -Force -ErrorAction SilentlyContinue
    }

    # Tidy up the leftovers from a manual attempt in the home folder.
    $strayVenv = Join-Path $HOME '.venv'
    if ((Test-Path (Join-Path $strayVenv 'pyvenv.cfg')) -and ($strayVenv -ne $venvDir)) {
        Write-Host ""
        Write-Note "Found an unused '.venv' folder in $HOME (from an earlier manual attempt)."
        if (Confirm-Default "Delete it?") {
            Remove-Item $strayVenv -Recurse -Force -ErrorAction SilentlyContinue
            Write-Ok "Deleted."
        }
    }
    $strayEnv = Join-Path $HOME '.env'
    if ((Test-Path $strayEnv) -and ((Get-Item $strayEnv).Length -eq 0)) {
        Remove-Item $strayEnv -Force -ErrorAction SilentlyContinue
    }

    Write-Host ""
    Write-Host "  All done! Start Suzy any time by double-clicking 'Suzy' on your desktop." -ForegroundColor Green
    Write-Host "  To update later, just run the same install command again." -ForegroundColor Green
    Write-Host ""
    if (Confirm-Default "Start Suzy now?") {
        Push-Location $InstallDir
        & $venvPython guard.py
        Pop-Location
    }
}

try {
    Install-Suzy
} catch {
    Write-Host ""
    Write-Host "  Setup stopped: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  Copy the messages above and send them over - we'll sort it out." -ForegroundColor Red
}
Write-Host ""
Read-Host "Press Enter to close" | Out-Null
