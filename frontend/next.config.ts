import type { NextConfig } from "next";

/**
 * Every API call the browser makes is same-origin (`/api/v1/...`) and is proxied here to
 * FastAPI. That keeps the `hl_session` cookie first-party, so it works unchanged in local
 * development and behind a single domain in production.
 */
// 127.0.0.1, not "localhost": on Windows localhost resolves to ::1 first, so anything else
// bound to IPv6 port 8000 (a WSL port-forward, another project's dev server) would swallow
// every API call and answer 404. The loopback literal always reaches our own backend.
const backendUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backendUrl}/api/:path*` }];
  },
};

export default nextConfig;
