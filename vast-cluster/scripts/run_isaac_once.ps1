Set-Location 'C:\Users\sande\cursor\SceneagentX'
$raw = & .\.venv\Scripts\vastai.exe show instances --raw
$info = $raw | & .\.venv\Scripts\python.exe -c "import json,sys; row=json.load(sys.stdin)[0]; print(row['id']); print(row['public_ipaddr']); print(row['ports']['22/tcp'][0]['HostPort']); print(row['cur_state'])"
$lines = @($info -split "`r?`n" | Where-Object { $_ -ne '' })
if ($lines[0] -ne '55073623' -or $lines[3] -ne 'running') {
    Write-Output ("INSTANCE " + $lines[0] + " " + $lines[3])
    exit 2
}
$ip = $lines[1]
$port = $lines[2]
Write-Output ("HOST " + $ip + " " + $port)
$key = Join-Path $env:USERPROFILE '.ssh\ssh_sceneagentX_1'
& scp -i $key -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $port vast-cluster\scripts\isaac_one_frame.sh root@${ip}:/root/isaac_one_frame.sh
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& ssh -i $key -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $port "root@${ip}" 'sed -i ''s/\r$//'' /root/isaac_one_frame.sh && bash /root/isaac_one_frame.sh'
exit $LASTEXITCODE
