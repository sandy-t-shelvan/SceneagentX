Set-Location 'C:\Users\sande\cursor\SceneagentX'
$raw = & .\.venv\Scripts\vastai.exe show instances --raw
$info = $raw | & .\.venv\Scripts\python.exe -c "import json,sys; row=json.load(sys.stdin)[0]; print(row['id']); print(row['public_ipaddr']); print(row['ports']['22/tcp'][0]['HostPort']); print(row['cur_state'])"
$lines = @($info -split "`r?`n" | Where-Object { $_ -ne '' })
if ($lines[0] -ne '55073623' -or $lines[3] -ne 'running') { exit 2 }
$ip = $lines[1]
$port = $lines[2]
$key = Join-Path $env:USERPROFILE '.ssh\ssh_sceneagentX_1'
& scp -i $key -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $port "root@${ip}:/root/SceneagentX/data/scan/isaac_check.json" data\scan\isaac_check.json
exit $LASTEXITCODE
