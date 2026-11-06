param([Parameter(ValueFromRemainingArguments = $true)][string[]]$RunArgs)
$ErrorActionPreference = 'Stop'
if (-not $env:GATEWAY_API_KEY) {
    $env:GATEWAY_API_KEY = [Environment]::GetEnvironmentVariable('GATEWAY_API_KEY', 'User')
}
if (-not $env:GATEWAY_API_KEY) { throw 'Set GATEWAY_API_KEY before running the agent.' }
uv run python @RunArgs
exit $LASTEXITCODE
