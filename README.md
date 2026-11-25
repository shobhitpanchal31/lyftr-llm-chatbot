# LLM Chatbot with Key Failover and Response Caching

A full-stack resilient AI chatbot built with **React**, **FastAPI**, **LiteLLM**, and **SQLite**, featuring ordered multi-key failover, LiteLLM response caching, per-request token tracking, and Dockerized multi-stage deployment.

## 1. Live URLs

- **Frontend URL:** `https://your-frontend-url.vercel.app` *(update after deploying)*
- **Backend URL:** `https://your-backend-url.onrender.com` *(update after deploying)*

## 2. Run locally

### With Docker (recommended)

1. Copy the environment template and fill in at least two real LLM API keys:
   ```bash
   cp .env.example .env
   ```
2. Start both services (backend on `http://localhost:8000`, frontend on `http://localhost:3000`):
   ```bash
   make up
   ```
   *(On Windows without `make`, run `docker compose up --build -d`.)*
3. Stop the stack:
   ```bash
   make down
   ```

### Without Docker

#### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export LLM_API_KEYS="key_one,key_two"
export LLM_MODEL="gemini/gemini-2.0-flash"
export DATABASE_URL="sqlite:///./app.db"
export ALLOWED_ORIGINS="http://localhost:3000"
export ENABLE_FAILURE_SIMULATION="true"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Run backend tests:
```bash
cd backend
pytest -v
```

#### Frontend
```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

## 3. Design decisions

- **Key fallback:** `LLM_API_KEYS` is parsed into an ordered pool (`key_1`, `key_2`, ...). On each live request, `LLMGateway` attempts keys sequentially via `litellm.completion`. When a provider raises a rate limit (`429`), quota exhaustion, `5xx`, or transient connection error (or a simulated failure via `X-Simulate-Failure`), the gateway increments `fallback_attempts` and immediately fails over to the next key without exposing raw API keys or stack traces. If every key fails, FastAPI returns `503 {"detail": "all providers unavailable"}` and nothing is written to SQLite.
- **Caching:** LiteLLM's built-in `Cache` (`local` in-memory by default, or `disk` via `CACHE_TYPE=disk`) is attached to `litellm.cache`. Cache keys are computed from the model and the full message list (`[system, ...history, user_message]`) while excluding `api_key`, so an identical first message in a new session resolves from `LLM_CACHE` regardless of which key originally answered, returning `key_id: null`, `fallback_attempts: 0`, and stored per-request token counts. When `X-Simulate-Failure` is active (with `ENABLE_FAILURE_SIMULATION=true`), cache lookup is bypassed (`no-cache: true, no-store: true`) so failover can always be demonstrated on demand.
- **SQLite session storage:** Conversation state lives exclusively in SQLite (`DATABASE_URL`, mounted at `/data/app.db` in Docker via a named volume) using WAL mode. The `messages` table stores `user` and `assistant` turns with assistant `metadata` serialized as JSON, and a `sessions` table tracks per-session `system_prompt` overrides. Failed requests (`422` or `503`) never write to SQLite.

## 4. Verification

### Health check
```bash
curl http://localhost:8000/health
```

### Cache hit across sessions
1. Send a first message in session `s-demo-1` (returns `resolution_source: "LIVE_INFERENCE"`):
   ```bash
   curl -s -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"session_id":"s-demo-1","message":"What is the capital of France?"}'
   ```
2. Send the exact same first message in a new session `s-demo-2` (returns `resolution_source: "LLM_CACHE"`, `key_id: null`, lower `latency_ms`):
   ```bash
   curl -s -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"session_id":"s-demo-2","message":"What is the capital of France?"}'
   ```

### Failover via `X-Simulate-Failure`
- Force `key_1` to return `429` (fails over to `key_2`, `fallback_attempts: 1`):
  ```bash
  curl -s -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -H "X-Simulate-Failure: key_1=429" \
    -d '{"session_id":"s-fail-1","message":"Hello after 429 failover"}'
  ```
- Force all keys to fail (returns `503 {"detail": "all providers unavailable"}` and adds nothing to history):
  ```bash
  curl -i -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -H "X-Simulate-Failure: *=429" \
    -d '{"session_id":"s-fail-all","message":"Will fail"}'
  ```

### Per-message token usage & multi-turn history
Send multiple turns with the same `session_id` and inspect `GET /sessions/{session_id}/messages`. Each assistant reply reports only that request's `input_tokens` (system + history + new message) and `output_tokens`.

## 5. Setup Used

- **Editor:** VS Code / Antigravity IDE
- **AI Tools:** Antigravity AI coding assistant for architecture, implementation, test generation, and Docker setup

## 6. Bonus features (if any)

- **UI Failover Simulation Selector:** Added an optional `Failover Hook` dropdown in the React UI header bar so `X-Simulate-Failure` (`key_1=429`, `key_1=500`, `*=429`) can be demonstrated visually in the browser alongside `curl`.
- **Runtime Frontend Config Injection:** The Nginx Docker image injects `VITE_API_BASE_URL` into `/config.js` at container startup, supporting both static build-time configuration (Vercel/Netlify) and Docker environment-variable configuration.
- **Per-Key Model Mapping:** Optional `LLM_MODELS` comma-separated variable allows pairing different LiteLLM model identifiers with different keys in `LLM_API_KEYS`.
