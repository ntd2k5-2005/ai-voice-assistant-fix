$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Venvs = Join-Path $Root ".venvs"
$Runtime = Get-Content (Join-Path $Root "services\tts\runtime.json") | ConvertFrom-Json
$VieNeuDir = Join-Path $Root ".runtime\VieNeu-TTS"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
$Uv = (Get-Command uv).Source
& $Uv python install 3.12
New-Item -ItemType Directory -Force -Path $Venvs | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root ".runtime") | Out-Null

foreach ($name in @("stt", "rag", "llm", "app")) {
    & $Uv venv --python 3.12 --seed (Join-Path $Venvs $name)
}
& "$Venvs\stt\Scripts\python.exe" -m pip install -r "$Root\services\stt\requirements.txt" -r "$Root\services\stt\requirements-test.txt"
& "$Venvs\rag\Scripts\python.exe" -m pip install -e "$Root\services\rag[api]"
& "$Venvs\llm\Scripts\python.exe" -m pip install -e "$Root\services\llm[api,test]"

if (-not (Test-Path (Join-Path $VieNeuDir ".git"))) {
    if (Test-Path $VieNeuDir) { Remove-Item -Recurse -Force $VieNeuDir }
    git clone $Runtime.repository $VieNeuDir
}
git -C $VieNeuDir fetch --depth 1 origin $Runtime.commit
git -C $VieNeuDir checkout --detach $Runtime.commit
$Actual = (git -C $VieNeuDir rev-parse HEAD).Trim()
if ($Actual -ne $Runtime.commit) { throw "VieNeu commit mismatch: expected=$($Runtime.commit) actual=$Actual" }
$TtsVenv = Join-Path $Venvs "tts"
if (Test-Path $TtsVenv) { Remove-Item -Recurse -Force $TtsVenv }
$env:UV_PROJECT_ENVIRONMENT = $TtsVenv
$env:UV_PYTHON = "3.12"
& $Uv sync --project $VieNeuDir --frozen
Remove-Item Env:UV_PROJECT_ENVIRONMENT
Remove-Item Env:UV_PYTHON
& $Uv pip install --python "$TtsVenv\Scripts\python.exe" "fastapi>=0.115,<1" "uvicorn[standard]>=0.30,<1" "pydantic>=2.8,<3" "pytest>=8,<10"

& "$Venvs\app\Scripts\python.exe" -m pip install -e "$Root\app[demo,dev]"
Write-Host ""
Write-Host "V10 environments installed. VieNeu pinned at $($Runtime.commit)."
Write-Host "Run: $Venvs\app\Scripts\python.exe $Root\scripts\start_services.py"
