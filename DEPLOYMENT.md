# Public Demo Deployment Checklist

This checklist is for deploying the public playable demo, especially on Streamlit Community Cloud.

## 1. Repository hygiene

- Confirm local-only files are ignored and not tracked:
  - `api_key.txt`
  - `openai_api_key.txt`
  - `llm_backend.txt`
  - `.streamlit/`
  - `.runtime/`
  - `.chroma/`
  - `*.db`
  - `__pycache__/`
- Do not commit provider API keys, Streamlit secrets, local databases, runtime session files, or uploaded player scripts.

## 2. Streamlit Cloud app settings

- Repository: this GitHub repository.
- Branch: the public demo branch.
- Main file path: `main.py`.
- Python version: use the version supported by Streamlit Cloud for this project.
- Install command: Streamlit Cloud should install `requirements.txt` automatically.
- Run command: Streamlit Cloud should run `streamlit run main.py` automatically.

## 3. Required secrets

Set these in the Streamlit Cloud app secrets UI, not in the repository.

```toml
OPENAI_API_KEY = "sk-..."
LLM_PROVIDER = "openai"
OPENAI_MODEL = "gpt-5.4-mini"
PUBLIC_DEMO_MODE = "true"
PUBLIC_DEMO_STRICT_CONFIG = "true"
PUBLIC_DEMO_ADMIN_DIAGNOSTICS = "false"
```

For NVIDIA/Qwen instead:

```toml
NVIDIA_API_KEY = "nvapi-..."
LLM_PROVIDER = "qwen"
NVIDIA_MODEL = "qwen/qwen2.5-7b-instruct"
PUBLIC_DEMO_MODE = "true"
PUBLIC_DEMO_STRICT_CONFIG = "true"
PUBLIC_DEMO_ADMIN_DIAGNOSTICS = "false"
```

## 4. Cost and abuse controls

Recommended starting values:

```toml
PUBLIC_DEMO_MAX_TURNS = "18"
PUBLIC_DEMO_MAX_INPUT_CHARS = "1200"
PUBLIC_DEMO_MAX_UPLOAD_BYTES = "200000"
PUBLIC_DEMO_MIN_TURN_SECONDS = "6"
PUBLIC_DEMO_DAILY_TURN_BUDGET = "250"
PUBLIC_DEMO_SESSION_TTL_HOURS = "24"
PUBLIC_DEMO_ADMIN_DIAGNOSTICS = "false"
```

For a multi-instance or more serious public deployment, add shared budget storage:

```toml
UPSTASH_REDIS_REST_URL = "https://..."
UPSTASH_REDIS_REST_TOKEN = "..."
```

Also set hard limits and alerts in the model provider dashboard. The app-level budget is a guardrail, not a billing guarantee.

## 5. Pre-deploy local checks

Run:

```bash
python -m py_compile app/ui.py app/vector_store.py test/test_public_demo_smoke.py
.venv/bin/python test/test_init.py
.venv/bin/python test/test_main.py
.venv/bin/python test/test_public_demo_smoke.py
```

## 6. Post-deploy smoke test

Open the deployed app and verify:

- The landing page loads without exposing debug prompts or stack traces.
- `Demo Story` starts successfully.
- `Upload Markdown` rejects oversized files with a friendly message.
- Character creation works with a preset.
- The first Keeper response is generated.
- Player input longer than `PUBLIC_DEMO_MAX_INPUT_CHARS` is blocked.
- Rapid repeated turns are throttled.
- The session stops at `PUBLIC_DEMO_MAX_TURNS`.
- The app never shows API keys, backend prompts, or raw HTML snippets.

## 7. Common failures

### I need to inspect deployment health

Temporarily set:

```toml
PUBLIC_DEMO_ADMIN_DIAGNOSTICS = "true"
```

The app will show a sidebar diagnostics panel with non-secret status: provider/model labels, credential presence, budget mode, local budget usage, runtime write access, embedding load state, and current story stage. Turn it off again before sharing the app widely.

### The app shows "not fully configured"

Check that a provider key is set in Streamlit secrets:

- `OPENAI_API_KEY`, or
- `NVIDIA_API_KEY`

Also check `LLM_PROVIDER` and model name spelling.

### The first request is slow

The landing page uses lazy embedding loading, but parsing and retrieval can still initialize model or Chroma resources after the player starts. This is expected on cold start.

### Daily budget resets unexpectedly

Without Upstash, the daily budget is stored locally in `.runtime/public_demo_usage.json`, which may reset when the app instance restarts. Use Upstash Redis for shared budget state.

### Provider costs rise too quickly

Lower:

- `PUBLIC_DEMO_MAX_TURNS`
- `PUBLIC_DEMO_DAILY_TURN_BUDGET`
- `OPENAI_MAX_TOKENS` or `NVIDIA_MAX_TOKENS`

Then add provider-side hard budget alerts.

## 8. Rollback

If the public app breaks:

- Temporarily set `PUBLIC_DEMO_STRICT_CONFIG = "true"` and remove the provider API key to show a friendly unavailable message.
- Or pause/delete the Streamlit Cloud deployment.
- Revert to the previous working commit in Streamlit Cloud.
