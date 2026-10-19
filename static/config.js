// Public backend origin only. Keep provider API keys on the Flask server.
// Flask and local development keep using their same-origin API.
// Branch-based GitHub Pages publishing uses the deployed Render backend.
// Actions builds can override this public URL with MATHSPHERE_API_URL.
window.MATHSPHERE_CONFIG = Object.freeze({
  apiBaseUrl: window.location.hostname.endsWith(".github.io")
    ? "https://mathsphere-web.onrender.com"
    : "",
});
