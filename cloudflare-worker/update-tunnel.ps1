param(
    [string]$WorkerUrl,
    [string]$TunnelUrl
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($WorkerUrl)) {
    $WorkerUrl = Read-Host "Permanent Worker URL (for example https://frame-permanent-link.account.workers.dev)"
}
if ([string]::IsNullOrWhiteSpace($TunnelUrl)) {
    $TunnelUrl = Read-Host "Current Cloudflare Quick Tunnel URL"
}

$workerUri = $null
if (
    -not [Uri]::TryCreate($WorkerUrl, [UriKind]::Absolute, [ref]$workerUri) -or
    $workerUri.Scheme -ne "https" -or
    -not [string]::IsNullOrEmpty($workerUri.UserInfo) -or
    -not [string]::IsNullOrEmpty($workerUri.Query) -or
    -not [string]::IsNullOrEmpty($workerUri.Fragment) -or
    $workerUri.AbsolutePath -ne "/"
) {
    throw "Worker URL must be an HTTPS origin without a path, query, or fragment."
}

$tunnelUri = $null
if (
    -not [Uri]::TryCreate($TunnelUrl, [UriKind]::Absolute, [ref]$tunnelUri) -or
    $tunnelUri.Scheme -ne "https" -or
    $tunnelUri.Host -notmatch '^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.trycloudflare\.com$' -or
    -not [string]::IsNullOrEmpty($tunnelUri.UserInfo) -or
    -not [string]::IsNullOrEmpty($tunnelUri.Query) -or
    -not [string]::IsNullOrEmpty($tunnelUri.Fragment) -or
    $tunnelUri.AbsolutePath -ne "/"
) {
    throw "Tunnel URL must be an HTTPS Quick Tunnel URL like https://random-name.trycloudflare.com."
}

$secureToken = Read-Host "Worker update token" -AsSecureString
$tokenPointer = [IntPtr]::Zero
$token = $null
try {
    $tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
    $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer)
    $headers = @{ Authorization = "Bearer $token" }
    $body = @{ url = $tunnelUri.GetLeftPart([UriPartial]::Authority) } | ConvertTo-Json -Compress
    $null = Invoke-RestMethod `
        -Method Post `
        -Uri "$($workerUri.AbsoluteUri.TrimEnd('/'))/_update" `
        -Headers $headers `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 20
    Write-Host "Redirect destination updated. Allow a short time for KV changes to propagate."
}
catch {
    throw "Tunnel update failed. Check the Worker URL, token, deployment, and KV binding. $($_.Exception.Message)"
}
finally {
    if ($tokenPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer)
    }
    $token = $null
    $secureToken.Dispose()
}
