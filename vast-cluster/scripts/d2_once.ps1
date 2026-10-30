$k = [Environment]::GetEnvironmentVariable('OPENAI_API_KEY', 'User')
if ([string]::IsNullOrWhiteSpace($k)) {
    Write-Output 'KEY_UNSET'
    exit 0
}
$env:OPENAI_API_KEY = $k
Set-Location 'C:\Users\sande\cursor\SceneagentX'
& .\.venv\Scripts\python.exe vast-cluster\scripts\d2_once.py
