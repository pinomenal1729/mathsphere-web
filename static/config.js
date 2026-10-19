// Public backend origin only. Keep provider API keys on the Flask server.
// Empty means same-origin Flask hosting. Pages builds can set MATHSPHERE_API_URL.
window.MATHSPHERE_CONFIG = Object.freeze({ apiBaseUrl: "" });
