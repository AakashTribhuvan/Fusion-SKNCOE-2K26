# Permanent FRAME link with Cloudflare Workers

This Worker gives FRAME a stable URL such as
`https://frame-permanent-link.<your-account-subdomain>.workers.dev`. It reads
the current Cloudflare Quick Tunnel URL from Workers KV and responds with a
`302 Found` redirect. It does not host FRAME or keep the tunnel running.

The Worker is deployed at
`https://frame-permanent-link.aakashtribhuvan2006.workers.dev`.

## What can go wrong

- The PC, FRAME API, and Quick Tunnel must all stay running. If any stops, the
  permanent link cannot make FRAME available.
- Quick Tunnel URLs change after a restart. Run the updater below after every
  restart, using the URL printed by `StartPrototype.bat`.
- The redirect sends visitors to the tunnel origin, not to a path on the
  Worker. Open the permanent link at its root.
- The app keeps sessions and credentials in process memory. Restarting FRAME
  loses them; existing sessions should be restarted after a tunnel or app
  restart.
- Workers KV is eventually consistent. A recently changed destination may
  take a short time to be observed everywhere. The Worker disables browser
  caching, but cannot make KV globally synchronous.
- The Worker accepts only HTTPS Quick Tunnel hosts matching
  `*.trycloudflare.com`. Named tunnels or custom hostnames need an intentional
  allowlist change in `cloudflare-worker/src/index.js`.
- Do not enable the passwordless `/admin` controls on a public demo tunnel.
  The Worker redirects to the app; it does not add authentication to FRAME.
- Worker and KV free-plan quotas and features can change. Check the current
  Cloudflare dashboard/plan before a high-traffic event. This design uses one
  small KV read per redirect and one KV write per tunnel update.

## One-time setup with Wrangler

Prerequisites: a Cloudflare account with Workers enabled, Node.js/npm, and
Wrangler. These commands run from the repository root in PowerShell:

```powershell
npx wrangler login
npx wrangler kv namespace create FRAME_LINKS
```

Copy the namespace ID printed by the second command into `id` in
`cloudflare-worker/wrangler.jsonc`, replacing
`REPLACE_WITH_KV_NAMESPACE_ID`. Do not commit credentials or tokens.

From the Worker directory, deploy and set its secret:

```powershell
Set-Location .\cloudflare-worker
npx wrangler deploy
npx wrangler secret put UPDATE_TOKEN
```

Generate a strong update token in PowerShell, copy it, and paste it into the
hidden `wrangler secret put` prompt. This stores it as a Worker secret rather
than in the Worker source or configuration:

```powershell
$bytes = New-Object byte[] 32
$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($bytes)
$token = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
Set-Clipboard $token
$rng.Dispose()
$token = $null
Remove-Variable bytes, rng, token
```

After deployment, Wrangler prints the `workers.dev` address. That is the
permanent URL to submit. Its exact account subdomain is assigned by Cloudflare;
the Worker name is `frame-permanent-link`.

The deploy will initially return `503` until a tunnel URL is configured. The
health endpoint is `<worker-url>/health`; it returns only whether a valid
destination is configured, never the destination or update token.

## Update after restarting the tunnel

Keep `StartPrototype.bat` running and copy the Quick Tunnel URL it prints.
From the repository root, run:

```powershell
.\cloudflare-worker\update-tunnel.ps1
```

Enter the permanent Worker URL and current tunnel URL when prompted, then
paste the update token into the masked prompt. You may also pass the two URLs
as parameters:

```powershell
.\cloudflare-worker\update-tunnel.ps1 `
  -WorkerUrl "https://frame-permanent-link.<your-account-subdomain>.workers.dev" `
  -TunnelUrl "https://random-name.trycloudflare.com"
```

The script still prompts for the token securely. Do not put the token in a
command argument, source file, or shell history. After updating, allow a short
KV propagation interval and test the permanent link.

## Endpoints and protection

- `GET /`: redirect to the KV destination with status `302`.
- `GET /health`: return `{ "configured": true|false }`; return `503` if missing
  or invalid. It never returns the destination.
- `POST /_update`: require the update token in the HTTP Authorization bearer
  scheme and a JSON body such as
  `{ "url": "https://random-name.trycloudflare.com" }`.
- The destination is validated both when written and when read. HTTP, user
  info, query strings, fragments, paths, non-default ports, and hosts outside
  the exact Quick Tunnel hostname pattern are rejected. Query parameters to
  `/` cannot change the redirect.

## Test plan

Run these checks after deployment:

1. Before the first update, open `/` and `/health`. Both should return `503`;
   the health body should say `configured: false`.
2. Run `update-tunnel.ps1` with a current tunnel URL. It should report success.
   After KV propagation, `/health` should return `200` and
   `configured: true`, without exposing a URL.
3. Inspect the permanent root response with `curl.exe -i <worker-url>`.
   Expect `302` and a `Location` header for the current tunnel. Adding a query
   such as `?url=https://example.com` must not change that location.
4. Restart the tunnel, update with its new URL, and retry after a short wait.
   The `Location` should change while the Worker URL remains the same.
5. Send an update without the token or with a wrong token. It should return
   `401`; the configured destination should remain unchanged.
6. With a valid token, try `http://random-name.trycloudflare.com`,
   `https://example.com`, a URL with credentials, or a URL with a path. Each
   should return `400`; the previous destination should remain unchanged.

The update endpoint accepts only JSON `POST`; wrong methods return `405`.
Malformed JSON returns `400`, and a missing KV destination returns `503`.
