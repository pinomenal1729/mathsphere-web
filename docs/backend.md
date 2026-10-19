# Backend deployment and AI troubleshooting

The interface's "AI tutor isn't connected" response previously covered several different failures, including rejected credentials, provider quotas, retired models, and SDK initialization failures. A healthy `/api/health` response with both AI service flags set to `true` only demonstrated client initialization; it did not verify an actual generation request.

## Render configuration

Use Python **3.11.8** with these commands:

```sh
# Build command
pip install -r requirements.txt

# Start command
gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

Remove any old `"httpx==0.27.2"` suffix from an existing Render build command. The updated Groq and Google SDKs use the HTTPX version pinned in `requirements.txt`; forcing the earlier version conflicts with the Google SDK's requirements.

Save API keys in the Render web service's **Environment** settings, not in GitHub Pages variables or frontend JavaScript. Existing keys can be reused:

- `GROQ_API_KEY` enables Groq.
- `GEMINI_API_KEY` enables Gemini. `GOOGLE_API_KEY` is also accepted as a fallback name.
- `GROQ_MODEL` optionally overrides `openai/gpt-oss-20b`, the text model used in Groq's current official SDK examples.
- `GEMINI_MODEL` optionally overrides `gemini-flash-latest`, the alias documented by the current official Google SDK.

Save settings and deploy the latest commit. Existing manually configured services must update their build command explicitly; changes to `render.yaml` alone may not update a manually created service.

## What changed

Google integration now uses the supported `google-genai` SDK and its HTTPS client for text and image requests, replacing `google-generativeai` and the hard-coded `gemini-2.0-flash` model. Groq is updated to a compatible SDK release, uses `openai/gpt-oss-20b` by default, and sends the current `max_completion_tokens` parameter.

Provider generation requests have 20-second timeouts. A model error or upstream HTTP 503 triggers a model-list request with a five-second timeout. The app may then try one different compatible model that the provider actually lists; it never guesses an unlisted model and never loops through the inventory. Text-model selection excludes audio, image-generation-only, embedding, and guard models. Successful replacements are reused for that worker's lifetime. Authentication, quota, empty-response, and timeout errors do not trigger model discovery. A failed Groq text request still falls back to Gemini. These limits bound recovery to at most two generation attempts per provider. Model availability does not guarantee that a generation request will succeed during a provider outage.

API failures return an appropriate non-200 status and a safe error message. The response includes a machine-readable `code` and provider failure categories, while keys and raw provider error payloads are excluded. Graph generation remains available when AI analysis fails.

| Error code                 | Action                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------ |
| `ai_not_configured`        | Add a provider key in the Render service environment, save, and redeploy.            |
| `ai_initialization_failed` | Deploy with the current requirements and inspect startup logs.                       |
| `ai_authentication_failed` | Check the corresponding key and project permissions with the provider.               |
| `ai_quota_exceeded`        | Check provider usage limits/billing or wait for the rate limit to reset.             |
| `ai_model_unavailable`     | Set the corresponding model variable to one available to that provider/project.      |
| `ai_request_rejected`      | Try a shorter question or another supported image.                                   |
| `ai_provider_unavailable`  | Retry after a temporary service or connectivity failure.                             |
| `ai_empty_response`        | Try a shorter question or a different model; the provider returned no usable answer. |
| `ai_timeout`               | Retry after the provider failed to respond within the request timeout.               |

`/api/health` includes `ai.providers` with configuration/client status and safe `last_error_code` fields, `ai.models` for configured models, and `ai.active_models` for models currently used after successful recovery. Client readiness is not proof of a successful upstream request. Last-error diagnostics and active model choices are per Gunicorn worker and may differ between requests.

## Verification

After deploying, send a short question such as "What is 2 + 2? Explain briefly". A substantive answer verifies generation; a healthy status alone does not. If generation fails, use the returned error category and the corresponding provider dashboard. Do not share secret values in chat or logs.

The backend regression tests instantiate the actual Groq and Google SDKs with in-memory HTTP transports. They verify successful generation, fallback, image serialization, authentication/quota/model errors, credential-safe diagnostics, and graph availability without making real provider calls:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_ai.py' -v
```
