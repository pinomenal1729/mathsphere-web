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
