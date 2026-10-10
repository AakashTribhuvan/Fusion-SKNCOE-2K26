const TARGET_KEY = "current-tunnel-url";
const UPDATE_PATH = "/_update";
const HEALTH_PATH = "/health";
const MAX_UPDATE_BODY_BYTES = 2048;

const JSON_HEADERS = {
  "Content-Type": "application/json; charset=utf-8",
  "Cache-Control": "no-store",
  "X-Content-Type-Options": "nosniff",
};

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: JSON_HEADERS,
  });
}

function validTunnelUrl(value) {
  if (typeof value !== "string" || value.length > 512) {
    return null;
  }

  let destination;
  try {
    destination = new URL(value);
  } catch {
    return null;
  }

  const labels = destination.hostname.toLowerCase().split(".");
  const tunnelName = labels[0] ?? "";
  const isQuickTunnel =
    labels.length === 3 &&
    labels[1] === "trycloudflare" &&
    labels[2] === "com" &&
    /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/.test(tunnelName);

  if (
    destination.protocol !== "https:" ||
    !isQuickTunnel ||
    destination.username ||
    destination.password ||
    destination.port ||
    destination.pathname !== "/" ||
    destination.search ||
    destination.hash
  ) {
    return null;
  }

  return destination.origin;
}

function constantTimeEqual(left, right) {
  const length = Math.max(left.length, right.length);
  let difference = left.length ^ right.length;
  for (let index = 0; index < length; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

async function tokenMatches(providedToken, expectedToken) {
  const encoder = new TextEncoder();
  const [providedHash, expectedHash] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(providedToken)),
    crypto.subtle.digest("SHA-256", encoder.encode(expectedToken)),
  ]);
  return constantTimeEqual(
    new Uint8Array(providedHash),
    new Uint8Array(expectedHash),
  );
}

async function updateDestination(request, env) {
  if (!env.UPDATE_TOKEN) {
    return jsonResponse({ error: "Update service is not configured." }, 503);
  }
  if (!env.FRAME_LINKS) {
    return jsonResponse({ error: "Destination storage is not configured." }, 503);
  }

  const authorization = request.headers.get("Authorization") ?? "";
  const match = /^Bearer\s+([^\s]+)$/i.exec(authorization);
  if (
    !match ||
    match[1].length > 512 ||
    !(await tokenMatches(match[1], env.UPDATE_TOKEN))
  ) {
    return jsonResponse({ error: "Unauthorized." }, 401);
  }

  const contentType = request.headers.get("Content-Type") ?? "";
  if (contentType.split(";", 1)[0].trim().toLowerCase() !== "application/json") {
    return jsonResponse({ error: "Content-Type must be application/json." }, 415);
  }

  const declaredLength = Number(request.headers.get("Content-Length") ?? 0);
  if (declaredLength > MAX_UPDATE_BODY_BYTES) {
    return jsonResponse({ error: "Request body is too large." }, 413);
  }

  const bodyText = await request.text();
  if (new TextEncoder().encode(bodyText).length > MAX_UPDATE_BODY_BYTES) {
    return jsonResponse({ error: "Request body is too large." }, 413);
  }

  let body;
  try {
    body = JSON.parse(bodyText);
  } catch {
    return jsonResponse({ error: "Request body must be valid JSON." }, 400);
  }

  const destination = validTunnelUrl(body?.url);
  if (!destination) {
    return jsonResponse({
      error: "url must be an HTTPS Cloudflare Quick Tunnel URL on trycloudflare.com.",
    }, 400);
  }

  await env.FRAME_LINKS.put(TARGET_KEY, destination);
  return jsonResponse({ updated: true });
}

async function health(env) {
  if (!env.FRAME_LINKS) {
    return jsonResponse({ configured: false }, 503);
  }

  const value = await env.FRAME_LINKS.get(TARGET_KEY);
  const configured = validTunnelUrl(value) !== null;
  return jsonResponse({ configured }, configured ? 200 : 503);
}

export default {
  async fetch(request, env) {
    const { pathname } = new URL(request.url);

    if (pathname === UPDATE_PATH) {
      if (request.method !== "POST") {
        return jsonResponse({ error: "Method not allowed." }, 405);
      }
      return updateDestination(request, env);
    }

    if (pathname === HEALTH_PATH) {
      if (request.method !== "GET") {
        return jsonResponse({ error: "Method not allowed." }, 405);
      }
      return health(env);
    }

    if (pathname !== "/") {
      return new Response("Not found.", {
        status: 404,
        headers: { "Cache-Control": "no-store" },
      });
    }

    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method not allowed.", {
        status: 405,
        headers: { Allow: "GET, HEAD", "Cache-Control": "no-store" },
      });
    }

    if (!env.FRAME_LINKS) {
      return new Response("FRAME link is not configured yet.", {
        status: 503,
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Cache-Control": "no-store",
        },
      });
    }

    const destination = validTunnelUrl(await env.FRAME_LINKS.get(TARGET_KEY));
    if (!destination) {
      return new Response("FRAME link is not configured yet.", {
        status: 503,
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Cache-Control": "no-store",
        },
      });
    }

    return new Response(null, {
      status: 302,
      headers: {
        Location: destination,
        "Cache-Control": "no-store",
      },
    });
  },
};
