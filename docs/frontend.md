# MathSphere frontend

The frontend is served directly by the existing Flask application. There is no frontend build step, Node.js runtime requirement, or external CDN dependency.

- `static/index.html`: application shell, navigation, and accessible dialogs.
- `static/css/app.css`: responsive layouts, local fonts, and the visual system.
- `static/js/app.js`: dashboard, client-side routing, API integration, graphs, conversations, and saved work.
- `static/vendor/katex`: KaTeX 0.16.11, its fonts, and MIT license.
- `static/fonts`: locally served DM Sans and Manrope variable fonts, with their licenses.

## Learning tools

All eight existing tools use the Flask endpoints: AI chat with optional image attachment, function plotting, the question bank, formula sheets, quizzes, theorem proofs, mathematician profiles, and project ideas. Exam preparation uses `/api/exam/info`.

Graphs render as SVG from the backend's computed values, support pointer inspection, and can be exported. They work without an AI provider. Question-bank results include individually revealable solutions and a clear empty state. Built-in mathematician profiles also work without AI.

AI requests require the existing server-side provider configuration. API keys are never entered or stored in the frontend. Missing providers, failed requests, validation errors, and rate limits produce explicit feedback rather than fabricated answers.

## Saved work

Conversations, saved results, preferences, daily-challenge answers, and recent activity are stored under `mathsphere.*` in the current browser's local storage. They do not sync across devices or constitute a user account. Images are sent with the current request and are excluded from persisted conversation history. The library supports searching, deletion, exporting all saved work as JSON, and downloading individual discoveries as Markdown.

## Interaction and accessibility

The layout adapts to mobile, tablet, and desktop widths. Mobile navigation has a backdrop, disables background scrolling, and removes the closed drawer from keyboard navigation. Dialogs use the native dialog focus behavior. Controls have accessible names, visible focus indicators, and keyboard shortcuts. Search opens with Ctrl/Cmd+K; chat sends with Enter and accepts a new line with Shift+Enter. Reduced-motion preferences are respected.

All server/user text is escaped before formatting. KaTeX runs with trusted commands disabled. External biography links are restricted to Wikipedia and open with `noopener noreferrer`.

## Validation

Validated using Chromium with Playwright at desktop (1440px) and mobile (390px) widths, including tool navigation, topic filters, daily-answer persistence, graph generation, empty database responses, saved profiles, library navigation, exam switching, chat/provider error handling, preferences, and mobile overflow. JavaScript syntax was checked with `node --check static/js/app.js`. Accessibility was checked with axe-core against WCAG 2 A/AA and WCAG 2.1 AA rules.

Live AI responses require configured provider credentials and were not validated during this change. The bundled question database is empty; the interface reports this honestly.

The committed browser regression suite uses deterministic API fixtures for AI response rendering, multiline equations, HTML escaping, image attachment handling, persistence, error states, search, and mobile navigation. It does not make live AI calls. With the Flask server running, install the browser test dependency outside the repository and run:

```sh
npm install --prefix /tmp/mathsphere-tests playwright@1.64.1
NODE_PATH=/tmp/mathsphere-tests/node_modules BROWSER_EXECUTABLE=/usr/bin/chromium node tests/frontend.cjs
```

Set `MATHSPHERE_URL` if the server is not on port 5000, or set `BROWSER_EXECUTABLE` to your local Chromium executable.

## Publishing to GitHub Pages

A project site such as `https://pinomenal1729.github.io/mathsphere-web/` needs a published `index.html` and assets that respect the `/mathsphere-web/` prefix. The frontend now uses relative asset URLs, including CSS font URLs.

The `Publish MathSphere frontend` workflow uploads `static/` as the Pages website root. In repository **Settings → Pages**, choose **GitHub Actions** as the source. The workflow runs for frontend changes on `main` and `feat/mathsphere-frontend`, and can also be started manually from Actions. If the `github-pages` environment has deployment branch restrictions, allow the branch being deployed.

For an existing **Deploy from a branch** setup, select the branch containing this frontend and the **/(root)** folder. The root `index.html` redirects to `static/`; `.nojekyll` keeps this a plain static site. Do not select `/docs`, which contains documentation rather than the website.

### Connect the Python backend

GitHub Pages serves the interface but cannot run Flask, SQLite API queries, or AI requests. For the complete application, deploy the Flask service using the existing Render configuration, or another Python host, and configure its server-side provider keys there.

The GitHub Pages site is configured to use **https://mathsphere-web.onrender.com** as its backend. Local development and Flask-hosted pages continue to use their same-origin `/api` endpoints.

For an Actions deployment, the workflow defaults to that Render URL. To use a different backend, set repository **Settings → Secrets and variables → Actions → Variables → MATHSPHERE_API_URL** to its public HTTPS origin. Use the backend origin without `/api`; the frontend appends each API path. The build validates and writes that public URL to `static/config.js`. This value is public configuration, not a secret. Never put an AI provider key in it.

For branch-based Pages publishing, `static/config.js` selects the Render URL on `*.github.io` and uses same-origin API requests elsewhere. To change backends or use a custom frontend domain, update that file as well. The existing Flask API already enables cross-origin access for `/api/*`.

Without a running backend, the dashboard, daily challenge, saved library, and local preferences still work. Server-powered tools display an availability message; a static deployment does not simulate working APIs.

The Pages regression suite starts its own local static server and checks both artifact and branch-based publishing, asset loading under a repository prefix, and external backend routing:

```sh
NODE_PATH=/tmp/mathsphere-tests/node_modules BROWSER_EXECUTABLE=/usr/bin/chromium node tests/pages.cjs
```

## Mathematical tutoring

Chat now supports step-by-step, concise, and exam-ready explanations. Independent calculation results and retrieved reference notes are displayed separately from model-written explanations and persist with saved conversations. See [Mathematical rigor and explainability](mathematical-accuracy.md) for the calculation scope, evaluation coverage, and limitations.
