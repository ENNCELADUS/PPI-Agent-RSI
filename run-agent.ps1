param([Parameter(ValueFromRemainingArguments = $true)][string[]]$RunArgs)
$ErrorActionPreference = 'Stop'
if (-not $env:SHANGHAITECH_API_KEY) {
    $env:SHANGHAITECH_API_KEY = [Environment]::GetEnvironmentVariable('SHANGHAITECH_API_KEY', 'User')
}
if (-not $env:SHANGHAITECH_API_KEY) { throw 'Set SHANGHAITECH_API_KEY before running the agent.' }
uv run python @RunArgs
exit $LASTEXITCODE
