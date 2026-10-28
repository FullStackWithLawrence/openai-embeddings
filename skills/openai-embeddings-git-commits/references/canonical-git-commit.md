# Canonical Git Commit

A real Smarter commit (`fix(passthrough)`, released in 0.17.4), in the required
format. What makes it a good one:

- The subject has a type and the Django app as its scope, and it says what the
  commit achieves.
- The first paragraph explains the problem as a user saw it, and how
  investigating it uncovered the real scope.
- Each area of the change gets a heading paragraph, then one bullet per change,
  with the reason for each non-obvious decision.
- The tests are described by what they prove, not just listed.
- It ends with the current status, so the reader knows what's known to be
  skipped and why.
- You are the author, and Lawrence McDaniel is the co-author.

```console
git commit --author="Claude <noreply@anthropic.com>" \
  -m "fix(passthrough): make the prompt passthrough work across providers, and test every template against every provider" \
  -m "The dashboard's prompt passthrough page (/dashboard/passthrough/) failed with OpenAI's current models: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead. Investigating this showed the page was never tested the way users run it. The test sent one hand-written fixture per provider and never tried the page's three request templates (Hello World, Message Roles, Function Call). Testing every combination found 15 failures out of 24 across 8 providers." \
  -m "Passthrough client (smarter/apps/provider/clients.py):" \
  -m "- New normalize_request_data(): for the openai provider only, renames max_tokens to max_completion_tokens before calling the API. OpenAI deprecated max_tokens in Chat Completions, and its reasoning models reject it. Other OpenAI-compatible providers may only accept max_tokens, so the rename is limited to OpenAI. If the caller also sends max_completion_tokens, that value wins.
- New create_chat_completion(): some OpenAI reasoning models only accept function tools in /v1/chat/completions when reasoning_effort is 'none'. When OpenAI rejects a request with tools for exactly this reason (param == reasoning_effort) and the caller did not set reasoning_effort, the client retries once with reasoning_effort='none' and logs a warning explaining what it did, why, and how to avoid the retry. It does not add the parameter up front, because older models (gpt-4o, gpt-4.1) reject reasoning_effort entirely." \
  -m "Request templates (react/packages/smarter-prompt-passthrough):" \
  -m "- The three templates moved from templates.tsx into a new templates.json. The React page and the backend tests now read the same file, so a template added there is automatically tested against every provider. templates.tsx and the TemplateSelector dropdown are both built from it, so the dropdown no longer hard-codes its options.
- Function Call used the legacy 'functions' field, which OpenAI replaced with 'tools' and which Google AI rejects. It now uses 'tools'.
- Message Roles and Function Call set a non-default temperature, which OpenAI's reasoning models reject. Both no longer set it." \
  -m "Tests:" \
  -m "- New test_passthrough_templates sends every template in templates.json to every active provider, building the request exactly as the page does: the provider's default model plus the template body, posted to the provider's passthrough endpoint. Each combination runs as its own subtest.
- Third-party providers are unreliable (expired keys, exhausted credits, rate limits, retired models, outages), so any error that comes back from the provider (any openai.APIError subclass) is logged at ERROR level with the provider's own message, and that combination is skipped. Errors raised in Smarter's own code still fail the test.
- New smarter/apps/provider/tests/test_clients.py covers the max_tokens rename and the reasoning_effort retry, including the cases where neither should happen." \
  -m "Provider setup (initialize_providers.py), checked live against each provider, including a request with tools:" \
  -m "- cohere: base_url changed to its OpenAI-compatible endpoint, https://api.cohere.ai/compatibility/v1/. The previous https://api.cohere.com/v1/ is Cohere's native API and returned 405.
- metaai: base_url is the SDK root that 'chat/completions' is appended to, not an endpoint URL.
- Fixed a crash when storing provider models: Together AI's /models endpoint returns a bare JSON list, not {\"data\": [...]}, which aborted the whole command." \
  -m "Current status: openai, googleai, cohere, fireworks, togetherai and metaai pass every template. anthropic (account credit balance too low) and mistral (429 rate limited, account-level) are skipped for account reasons, not code defects." \
  -m "Co-authored-by: Lawrence McDaniel <lpm0073@gmail.com>"
```

Notes on the format:

- A bullet list goes in a single `-m` with real newlines between bullets. A
  paragraph gets its own `-m`.
- Inner double quotes are escaped (`\"data\"`). Backticks are avoided
  altogether.
- This subject is longer than 72 characters. That passed because the
  commit-msg hook isn't installed everywhere. Prefer a header within 72.
