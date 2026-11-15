import type { NextConfig } from "next";

/**
 * Every API call the browser makes is same-origin (`/api/v1/...`) and is proxied here to
 * FastAPI. That keeps the `hl_session` cookie first-party, so it works unchanged in local
 * development and behind a single domain in production.
 */
const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backendUrl}/api/:path*` }];
  },
};

export default nextConfig;
