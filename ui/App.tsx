// src/App.tsx — Chat + Upload + Image Rendering (Bootstrap, ES6+)
import React, { useEffect, useMemo, useRef, useState } from "react";

type Role = "user" | "assistant" | "system";

interface ChatMessage {
  role: Role;
  content: string;
}

interface UploadedMeta {
  id: string;
  name: string;
  url?: string;
  size: number;
  type: string;
}

const API_BASE = "http://localhost:8000";
const CHAT_URL = `${API_BASE}/chat`;
const UPLOAD_URL = `${API_BASE}/upload`;

// Format file sizes
const humanSize = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
};

// Render avatars
const Avatar: React.FC<{ role: Role }> = ({ role }) => {
  const isAssistant = role === "assistant";
  const cls = `d-flex align-items-center justify-content-center rounded-circle fw-semibold ${
    isAssistant ? "bg-primary text-white" : "bg-secondary-subtle text-body"
  }`;
  return (
    <div className={cls} style={{ width: 32, height: 32, fontSize: 11 }}>
      {isAssistant ? "AI" : "You"}
    </div>
  );
};

// 🧩 Utility to detect and render images inside messages
const renderMessageContent = (text: string) => {
  const imgRegex = /(sandbox:\/files\/[^\s)]+\.png|\/files\/[^\s)]+\.png|https?:\/\/[^\s)]+\.png)/gi;
  const parts = text.split(imgRegex);

  return parts.map((part, i) => {
    const trimmed = part.trim();
    if (trimmed.match(imgRegex)) {
      const src = trimmed.replace(/^sandbox:/, API_BASE);
      return (
        <div key={i} className="my-2 text-center">
          <img
            src={src}
            alt="AI-generated"
            className="img-fluid rounded shadow-sm"
            style={{ maxHeight: "320px", objectFit: "contain" }}
          />
        </div>
      );
    }
    return <span key={i}>{part}</span>;
  });
};

// Chat message bubble
const MessageBubble: React.FC<{ message: ChatMessage }> = ({ message }) => {
  const isAssistant = message.role === "assistant";
  return (
    <div
      className={`d-flex w-100 gap-2 ${
        isAssistant ? "justify-content-start" : "justify-content-end"
      }`}
    >
      {isAssistant && <Avatar role={message.role} />}
      <div
        className={`p-3 rounded-4 shadow-sm ${
          isAssistant ? "bg-white border" : "bg-primary text-white"
        }`}
        style={{
          maxWidth: "80%",
          whiteSpace: "pre-wrap",
          lineHeight: 1.6,
          fontSize: "0.95rem",
          color: isAssistant ? "#2c2c2c" : "#fff",
        }}
      >
        {renderMessageContent(message.content)}
      </div>
      {!isAssistant && <Avatar role={message.role} />}
    </div>
  );
};

// Small file pill display
const FilePill: React.FC<{ f: UploadedMeta; onRemove: () => void }> = ({
  f,
  onRemove,
}) => (
  <span
    className="badge d-inline-flex align-items-center gap-2 me-2 mb-2"
    style={{
      backgroundColor: "#f1f3f5",
      color: "#344054",
      borderRadius: 999,
      padding: "0.5rem 0.75rem",
    }}
  >
    <span className="text-truncate" style={{ maxWidth: 220 }}>
      {f.name}
    </span>
    <button
      className="btn btn-sm btn-link p-0"
      type="button"
      aria-label="Remove"
      onClick={onRemove}
      style={{ textDecoration: "none", color: "#667085" }}
      title="Detach file"
    >
      ×
    </button>
  </span>
);

// ===== Main App =====
const App: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "👋 Hi! Upload a CSV/XLSX file, then ask me anything about it — I'll analyze it directly in this chat.",
    },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const logRef = useRef<HTMLDivElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const [uploads, setUploads] = useState<UploadedMeta[]>([]);
  const [attachedNext, setAttachedNext] = useState<UploadedMeta[]>([]);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    if (logRef.current)
      logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [messages, busy]);

  const canSend = useMemo(() => input.trim().length > 0 && !busy, [input, busy]);

  // Upload files → auto attach to next message
  const onChooseFiles: React.ChangeEventHandler<HTMLInputElement> = async (e) => {
    const files = e.target.files ? Array.from(e.target.files) : [];
    if (!files.length) return;
    e.currentTarget.value = "";
    await doUpload(files);
  };

  const doUpload = async (files: File[]) => {
    setUploading(true);
    setError("");
    try {
      const form = new FormData();
      files.forEach((f) => form.append("files", f, f.name));
      const res = await fetch(UPLOAD_URL, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Upload failed (HTTP ${res.status})`);
      const data = (await res.json()) as UploadedMeta[];
      setUploads((prev) => [...data, ...prev]);
      setAttachedNext((prev) => [...prev, ...data]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  const detach = (id: string) =>
    setAttachedNext((prev) => prev.filter((f) => f.id !== id));

  const buildAttachmentsBlock = (files: UploadedMeta[]) => {
    if (!files.length) return "";
    const lines = files.map(
      (u, i) => `#${i + 1}: ${u.name}${u.url ? ` → ${u.url}` : ""} (id:${u.id})`
    );
    return `\n\n[Attachments]\n${lines.join("\n")}`;
  };

  const send = async (): Promise<void> => {
    if (!canSend) return;
    const attachmentsText = buildAttachmentsBlock(attachedNext);
    const userMsg: ChatMessage = {
      role: "user",
      content: input.trim() + attachmentsText,
    };
    const thread = [...messages, userMsg];
    setMessages(thread);
    setInput("");
    setBusy(true);
    setError("");
    setAttachedNext([]);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(CHAT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: thread.map(({ role, content }) => ({ role, content })),
        }),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as { reply?: string };
      const reply = data.reply ?? "(no reply)";
      setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "⚠️ Sorry, I hit an error talking to the backend. Check logs.",
        },
      ]);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  const stop = () => {
    if (abortRef.current) abortRef.current.abort();
    setBusy(false);
  };

  const onKeyDown: React.KeyboardEventHandler<HTMLTextAreaElement> = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  };

  const newChat = () => {
    setMessages([
      {
        role: "assistant",
        content:
          "🧠 New chat started. Upload a dataset and ask your question!",
      },
    ]);
    setAttachedNext([]);
  };

  // ============ UI ============
  return (
    <div className="d-flex flex-column vh-100" style={{ backgroundColor: "#f7f8fb" }}>
      {/* Header */}
      <nav
        className="navbar sticky-top shadow-sm"
        style={{
          background: "linear-gradient(90deg, #ffffff 0%, #f4f7ff 100%)",
          borderBottom: "1px solid #eaeaea",
        }}
      >
        <div className="container-fluid d-flex justify-content-between">
          <span className="navbar-brand mb-0 h6">ML Agentic Assistant</span>
          <button className="btn btn-outline-secondary" onClick={newChat}>
            New chat
          </button>
        </div>
      </nav>

      {/* Main area */}
      <main className="flex-grow-1 overflow-auto" ref={logRef}>
        <div className="container py-4">
          <div className="d-flex flex-column gap-3">
            {messages.map((m, i) => (
              <MessageBubble key={`${i}-${m.role}`} message={m} />
            ))}
            {busy && <div className="text-muted small">Assistant is thinking…</div>}
          </div>
        </div>
      </main>

      {/* Composer */}
      <div className="border-top bg-light">
        <div className="container py-3">
          <div className="card shadow-sm border-0 rounded-4">
            <div className="card-body bg-white">
              {attachedNext.length > 0 && (
                <>
                  <div className="small text-muted mb-1">Attached:</div>
                  <div className="mb-2">
                    {attachedNext.map((f) => (
                      <FilePill key={f.id} f={f} onRemove={() => detach(f.id)} />
                    ))}
                  </div>
                </>
              )}

              <div className="d-flex gap-2 mb-2">
                <label className="btn btn-outline-secondary mb-0">
                  Attach files
                  <input type="file" multiple className="d-none" onChange={onChooseFiles} />
                </label>
                {uploading && <span className="text-muted small">Uploading…</span>}
              </div>

              <textarea
                className="form-control mb-3"
                rows={3}
                placeholder="Ask me about your dataset..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
              />

              <div className="d-flex justify-content-between align-items-center">
                <div>{busy && <button onClick={stop} className="btn btn-outline-secondary">Stop</button>}</div>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => void send()}
                  disabled={!canSend}
                >
                  Send
                </button>
              </div>
              <div className="form-text mt-3">
                AI-generated insights may include plots or text — verify critical info.
              </div>
              {error && (
                <div className="alert alert-danger mt-3 mb-0 py-2 small" role="alert">
                  {error}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
