# Antigravity prompt enhancement

The image engine runs locally. Prompt enhancement uses Google Account sign-in
through Antigravity. No local prompt-enhancement LLM is loaded or downloaded.
The required Qwen3-VL image text encoder remains part of local image inference.

## Implementation sources inspected on 2026-09-21

- `C:/Users/ido/REPOS/moise/dashboard.html`: `connectAntigravity`,
  `submitAntigravityPaste`, `disconnectAntigravity`, connection status and polling.
- `C:/Users/ido/REPOS/moise/index.js`: `/api/oauth/antigravity/*`,
  `startAntigravityOAuthFlow`, callback listener, exchange, token refresh.
- `C:/Users/ido/REPOS/weboss/services/antigravityImageAuthService.ts` and
  `components/ImageModelDropdown.tsx`: popup, status, automatic callback,
  manual callback recovery and disconnect.
- `C:/Users/ido/REPOS/weboss/server.js`: text wire profiles,
  `applyAntigravityTextRequestProfile`, `callGeminiAntigravityTextOnce`.
- `C:/Users/ido/REPOS/docs-oauth/antigravity-oauth-llm-integration-guide.md`,
  `WEBOSS_AUTH_RUNTIME_CONTRACT.md`, `WEBOSS_AUTH_PARITY_MATRIX.md` and
  `antigravity-oauth-image-integration-guide.md`.

The current Weboss source supersedes stale placeholder IDs in the guide:
`MODEL_PLACEHOLDER_*` values belong in `request.labels.model_enum`, never in
the top-level model field. Generation sends the human wire ID, trajectory
labels, model-specific thinking configuration and CloudCode SSE envelope.

## Connection behavior

The local routes match the sibling apps: start, status, poll, exchange, refresh,
disconnect. The callback is `http://localhost:1455/auth/callback`, with PKCE
S256, random expiring state and automatic server-side exchange. If another
local app owns port 1455, the settings panel exposes full callback URL paste.

Tokens are encrypted with Windows DPAPI for the current Windows user, under
the data directory's `connections/antigravity.dpapi`. Browser state, job
requests, image metadata and logs contain no bearer or refresh token.
The status route returns connection state, account email and model selection.
Workers resolve and refresh the saved connection immediately before execution.
Refresh starts within five minutes of expiry, re-discovers the CloudCode project
pool and preserves an omitted replacement refresh token. An interprocess lock
prevents competing refreshes. Disconnect removes only this app's connection.

## Prompt contract

The small official Qwen T2I/I2I system prompt files are copied into
`reference/prompts/` with their source revisions in the model catalog. This
does not depend on retaining the local prompt-enhancer weight directories.
Up to ten input images are attached in order as PNG with `<imageN>` labels;
each preview is scaled to fit 1024 square pixels while preserving its aspect.
Output parsing preserves `rewritten_prompt`, `wh_ratio`, and `ratio_follow`.
The Qwen alpha-channel instruction survives rewriting.

With Enhance prompt off, no OAuth or cloud call is made. With it on, the prompt
and input-image previews are sent to Google, as stated beside the connection.
There is no silent switch to another provider or local LLM on failure.

## Validation boundary

The automated OAuth tests exercise DPAPI, PKCE/state/expiry, a real local HTTP
callback, refresh rotation, model wire labels, SSE/thought filtering, ordered
image inputs and transparency preservation. Google token exchange and text
responses in those tests are mocked. Live account sign-in and a successful
cloud rewrite must be verified separately before claiming provider readiness.
