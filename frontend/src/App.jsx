import React, { useEffect, useRef, useState } from 'react';

function getApiBaseUrl() {
  const runtimeUrl = window.__APP_CONFIG__?.API_BASE_URL;
  if (runtimeUrl && runtimeUrl.trim() !== '') {
    return runtimeUrl.replace(/\/+$/, '');
  }
  const buildUrl = import.meta.env.VITE_API_BASE_URL;
  if (buildUrl && buildUrl.trim() !== '') {
    return buildUrl.replace(/\/+$/, '');
  }
  return 'http://localhost:8000';
}

function generateSessionId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `s-${crypto.randomUUID().slice(0, 8)}`;
  }
  return `s-${Math.random().toString(36).slice(2, 10)}`;
}

function formatErrorDetail(detail, status) {
  if (typeof detail === 'string') {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const loc = Array.isArray(item.loc) ? item.loc.join('.') : '';
        return loc ? `${loc}: ${item.msg}` : item.msg || JSON.stringify(item);
      })
      .join('; ');
  }
  if (detail && typeof detail === 'object') {
    return JSON.stringify(detail);
  }
  return `Request failed with status ${status}`;
}

export default function App() {
  const apiBaseUrl = getApiBaseUrl();
  const [sessionId, setSessionId] = useState(() => generateSessionId());
  const [systemPrompt, setSystemPrompt] = useState('');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isSending, setIsSending] = useState(false);
  const [health, setHealth] = useState(null);
  const [simulateFailure, setSimulateFailure] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    let active = true;
    fetch(`${apiBaseUrl}/health`)
      .then(async (res) => {
        const data = await res.json().catch(() => null);
        if (active && res.ok && data) {
          setHealth(data);
        } else if (active) {
          setHealth({ status: 'degraded' });
        }
      })
      .catch(() => {
        if (active) {
          setHealth({ status: 'unreachable' });
        }
      });
    return () => {
      active = false;
    };
  }, [apiBaseUrl]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isSending]);

  function handleNewSession() {
    setSessionId(generateSessionId());
    setMessages([]);
    setInput('');
  }

  async function sendMessage() {
    const trimmed = input.trim();
    if (isSending) return;

    // Clear input right on send so chat-input is empty after sending
    setInput('');

    // Optimistically show user message only if non-empty; if empty, let backend return 422 or show validation error
    if (trimmed.length > 0) {
      setMessages((prev) => [
        ...prev,
        {
          id: `user-${Date.now()}-${Math.random()}`,
          role: 'user',
          content: trimmed,
        },
      ]);
    }

    setIsSending(true);
    try {
      const payload = {
        session_id: sessionId,
        message: trimmed,
      };
      if (systemPrompt.trim() !== '') {
        payload.system_prompt = systemPrompt.trim();
      }

      const headers = {
        'Content-Type': 'application/json',
      };
      if (simulateFailure.trim() !== '') {
        headers['X-Simulate-Failure'] = simulateFailure.trim();
      }

      const response = await fetch(`${apiBaseUrl}/chat`, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        const errorText = formatErrorDetail(data.detail, response.status);
        setMessages((prev) => [
          ...prev,
          {
            id: `error-${Date.now()}-${Math.random()}`,
            role: 'error',
            status: response.status,
            content: errorText,
          },
        ]);
        return;
      }

      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-${Date.now()}-${Math.random()}`,
          role: 'assistant',
          content: data.reply,
          metadata: data.metadata,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}-${Math.random()}`,
          role: 'error',
          status: 0,
          content: err?.message || 'Network error while contacting backend.',
        },
      ]);
    } finally {
      setIsSending(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    sendMessage();
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="header-left">
          <div className="brand-mark">L</div>
          <div>
            <h1>Lyftr AI Gateway</h1>
            <p className="subtitle">Multi-key failover &amp; LiteLLM response caching</p>
          </div>
        </div>

        <div className="header-right">
          {health && (
            <span
              className={`status-pill ${
                health.status === 'ok' ? 'status-ok' : 'status-warn'
              }`}
              title={`Backend health: ${health.status}`}
            >
              <span className="status-dot" />
              {health.status === 'ok'
                ? `${health.keys_configured} keys • ${
                    health.cache_enabled ? 'cache on' : 'cache off'
                  }`
                : `Backend: ${health.status}`}
            </span>
          )}
          <span className="session-badge" title="Current session ID">
            Session: <code>{sessionId}</code>
          </span>
          <button
            type="button"
            className="secondary-btn"
            onClick={handleNewSession}
            disabled={isSending}
          >
            New Session
          </button>
        </div>
      </header>

      <section className="controls-bar">
        <div className="control-group system-prompt-group">
          <label htmlFor="system-prompt-input">System Prompt</label>
          <input
            id="system-prompt-input"
            data-testid="system-prompt-input"
            type="text"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="Optional system prompt for this session (e.g., Answer in one concise sentence)"
          />
        </div>

        <div className="control-group simulation-group">
          <label htmlFor="simulate-failure-select">Failover Hook</label>
          <select
            id="simulate-failure-select"
            value={simulateFailure}
            onChange={(e) => setSimulateFailure(e.target.value)}
          >
            <option value="">Normal (no simulated failure)</option>
            <option value="key_1=429">Simulate key_1=429 (rate limit)</option>
            <option value="key_1=500">Simulate key_1=500 (server error)</option>
            <option value="*=429">Simulate *=429 (all keys fail → 503)</option>
          </select>
        </div>
      </section>

      <main className="chat-feed" aria-live="polite">
        {messages.length === 0 && (
          <div className="empty-state">
            <h2>Start a conversation</h2>
            <p>
              Messages are stored per session in SQLite. Repeat an identical prompt
              in a new session to see <code>LLM_CACHE</code>, or trigger{' '}
              <code>X-Simulate-Failure</code> to test automatic key failover.
            </p>
          </div>
        )}

        {messages.map((msg) => {
          if (msg.role === 'user') {
            return (
              <div key={msg.id} className="message-row message-row-user">
                <div className="message-bubble user-bubble">
                  <div className="message-role">You</div>
                  <div className="message-text">{msg.content}</div>
                </div>
              </div>
            );
          }

          if (msg.role === 'error') {
            return (
              <div key={msg.id} className="message-row message-row-error">
                <div className="message-bubble error-bubble" role="alert">
                  <div className="message-role">
                    Error {msg.status ? `(${msg.status})` : ''}
                  </div>
                  <div className="message-text">{msg.content}</div>
                </div>
              </div>
            );
          }

          const meta = msg.metadata || {};
          const isCached = meta.resolution_source === 'LLM_CACHE';

          return (
            <div key={msg.id} className="message-row message-row-assistant">
              <div
                className="message-bubble assistant-bubble"
                data-testid="assistant-message"
              >
                <div className="message-role">Assistant</div>
                <div className="message-text">{msg.content}</div>

                <div className="metadata-bar">
                  <span
                    className={`meta-badge ${
                      isCached ? 'meta-badge-cache' : 'meta-badge-live'
                    }`}
                    data-testid="meta-source"
                  >
                    {meta.resolution_source}
                  </span>

                  <span className="meta-chip">
                    <span className="meta-label">Model:</span>
                    <span data-testid="meta-model">{meta.model}</span>
                  </span>

                  <span className="meta-chip">
                    <span className="meta-label">Latency:</span>
                    <span data-testid="meta-latency">{meta.latency_ms} ms</span>
                  </span>

                  <span className="meta-chip">
                    <span className="meta-label">In:</span>
                    <span data-testid="meta-input-tokens">{meta.input_tokens}</span>
                  </span>

                  <span className="meta-chip">
                    <span className="meta-label">Out:</span>
                    <span data-testid="meta-output-tokens">{meta.output_tokens}</span>
                  </span>

                  <span className="meta-chip">
                    <span className="meta-label">Key:</span>
                    <span>{meta.key_id ?? 'null'}</span>
                  </span>

                  {typeof meta.fallback_attempts === 'number' &&
                    meta.fallback_attempts > 0 && (
                      <span className="meta-chip meta-chip-failover">
                        Fallbacks: {meta.fallback_attempts}
                      </span>
                    )}
                </div>
              </div>
            </div>
          );
        })}

        {isSending && (
          <div className="message-row message-row-assistant">
            <div className="message-bubble assistant-bubble typing-bubble">
              <span className="typing-dots">Thinking...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </main>

      <form className="composer" onSubmit={handleSubmit}>
        <textarea
          data-testid="chat-input"
          rows={2}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
        />
        <button
          data-testid="send-button"
          type="submit"
          className="primary-btn"
          disabled={isSending}
        >
          {isSending ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
}
