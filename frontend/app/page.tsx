// frontend/app/page.tsx
"use client";
// "use client" tells Next.js this component runs in the browser, not on the server.
// We need this because we use useState, useEffect, and browser APIs like fetch.

import { useState, useRef, useEffect } from "react";

// TypeScript type for a chat message.
// Union type means role can only ever be "user" or "assistant" — nothing else.
type Message = {
  role: "user" | "assistant";
  content: string;
};

// The URL of our FastAPI backend.
// In development this is localhost. When we deploy, we'll change this to
// the real server URL via an environment variable.
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  // useState holds the list of messages in the conversation.
  // Every time setMessages is called, React re-renders the component.
  const [messages, setMessages]   = useState<Message[]>([]);

  // The current text in the input box
  const [input, setInput]         = useState("");

  // Whether the AI is currently generating a response.
  // We use this to disable the send button and show a loading state.
  const [isStreaming, setIsStreaming] = useState(false);

  // Whether the backend is reachable. We check this on page load.
  const [serverReady, setServerReady] = useState<boolean | null>(null);

  // List of source documents currently indexed, shown in the sidebar.
  const [sources, setSources]     = useState<string[]>([]);

  // A ref to the bottom of the message list.
  // We use this to auto-scroll when new messages arrive.
  // useRef doesn't cause re-renders — it's just a pointer to a DOM element.
  const bottomRef = useRef<HTMLDivElement>(null);

  // useEffect runs after the component first renders (empty dependency array []).
  // We use it to check the backend health and load the sources list.
  useEffect(() => {
    checkServer();
    loadSources();
  }, []);

  // Auto-scroll to bottom whenever messages change.
  // The dependency array [messages] means this runs every time messages updates.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function checkServer() {
    // Verify the FastAPI server is reachable before the user tries to ask anything.
    try {
      const res = await fetch(`${API_URL}/health`);
      setServerReady(res.ok);
    } catch {
      // fetch throws if the server is completely unreachable (not just a bad status)
      setServerReady(false);
    }
  }

  async function loadSources() {
    try {
      const res  = await fetch(`${API_URL}/sources`);
      const data = await res.json();
      // Strip file extensions for cleaner display in the sidebar
      setSources(data.sources.map((s: string) => s.replace(/\.pdf$/i, "")));
    } catch {
      // Sources are cosmetic — silently fail if the server isn't ready yet
    }
  }

  async function sendMessage() {
    // Don't send if input is empty or we're already streaming
    if (!input.trim() || isStreaming) return;

    const question = input.trim();
    setInput("");      // clear the input box immediately
    setIsStreaming(true);

    // Add the user's message to the conversation
    setMessages(prev => [...prev, { role: "user", content: question }]);

    // Add an empty assistant message that we'll fill in as tokens stream in.
    // This creates the "typing" effect — the message exists but is empty at first.
    setMessages(prev => [...prev, { role: "assistant", content: "" }]);

    try {
      // POST to our FastAPI /ask endpoint
      const response = await fetch(`${API_URL}/ask`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ question }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      // response.body is a ReadableStream — raw bytes arriving over the network.
      // We need a reader to consume it chunk by chunk.
      const reader  = response.body!.getReader();
      const decoder = new TextDecoder(); // converts bytes → string
      let   buffer  = "";               // accumulates partial lines between chunks

      // Read the stream in a loop until it's done
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Decode the raw bytes into a string and add to our buffer.
        // stream: true means "there might be more data coming, don't flush yet"
        buffer += decoder.decode(value, { stream: true });

        // Split on double newlines — that's the SSE event delimiter.
        // Each event looks like: "data: {...}\n\n"
        const lines = buffer.split("\n\n");

        // The last element might be an incomplete event (no closing \n\n yet).
        // Keep it in the buffer and process only the complete events.
        buffer = lines.pop() || "";

        for (const line of lines) {
          // Each SSE event starts with "data: "
          if (!line.startsWith("data: ")) continue;

          const payload = line.slice(6); // remove the "data: " prefix
          if (payload === "[DONE]") break;

          try {
            const parsed = JSON.parse(payload);

            if (parsed.text) {
              // Append the new token to the last message (the assistant's)
              // We use a functional update to always work with the latest state
              setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  ...updated[updated.length - 1],
                  content: updated[updated.length - 1].content + parsed.text,
                };
                return updated;
              });
            }

            if (parsed.error) {
              setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1].content = `Error: ${parsed.error}`;
                return updated;
              });
            }
          } catch {
            // JSON.parse failed — skip malformed events
          }
        }
      }
    } catch (err) {
      // Network error or server down — update the assistant message with the error
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1].content =
          "Failed to reach the server. Make sure the backend is running on port 8000.";
        return updated;
      });
    } finally {
      // Always re-enable the input when streaming ends, even if there was an error
      setIsStreaming(false);
    }
  }

  // Send on Enter key (but not Shift+Enter, which adds a newline)
  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault(); // stop the Enter from adding a newline in the textarea
      sendMessage();
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">

      {/* ── Sidebar ──────────────────────────────────────────────────────── */}
      <div className="w-64 bg-white border-r border-gray-200 flex flex-col p-4">
        <h1 className="text-lg font-semibold text-gray-900 mb-1">RAG Research Assistant</h1>
        <p className="text-xs text-gray-500 mb-6">Ask questions about your documents</p>

        {/* Server status indicator */}
        <div className="flex items-center gap-2 mb-6">
          <div className={`w-2 h-2 rounded-full ${
            serverReady === null  ? "bg-yellow-400" :
            serverReady           ? "bg-green-500"  : "bg-red-500"
          }`} />
          <span className="text-xs text-gray-500">
            {serverReady === null ? "Checking..." :
             serverReady          ? "Backend connected" : "Backend offline"}
          </span>
        </div>

        {/* Indexed documents list */}
        {sources.length > 0 && (
          <div>
            <p className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">
              Indexed documents
            </p>
            <ul className="space-y-1">
              {sources.map(src => (
                <li key={src}
                    className="text-xs text-gray-600 bg-gray-50 rounded px-2 py-1.5 truncate"
                    title={src}>
                  {src}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Clear chat button */}
        {messages.length > 0 && (
          <button
            onClick={() => setMessages([])}
            className="mt-auto text-xs text-gray-400 hover:text-gray-600 text-left">
            Clear conversation
          </button>
        )}
      </div>

      {/* ── Main chat area ────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col">

        {/* Message list */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">

          {/* Empty state shown before any messages */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h2 className="text-lg font-medium text-gray-700 mb-2">Ask your documents anything</h2>
              <p className="text-sm text-gray-400 max-w-sm">
                Questions are answered using only the indexed research papers — no hallucination from outside sources.
              </p>
            </div>
          )}

          {/* Render each message */}
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-2xl px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
                msg.role === "user"
                  ? "bg-blue-600 text-white rounded-br-sm"
                  : "bg-white border border-gray-200 text-gray-800 rounded-bl-sm"
              }`}>
                {/* Show a blinking cursor while the assistant is typing */}
                {msg.content || (isStreaming && i === messages.length - 1
                  ? <span className="inline-block w-2 h-4 bg-gray-400 animate-pulse" />
                  : ""
                )}
              </div>
            </div>
          ))}

          {/* Invisible div at the bottom — we scroll to this */}
          <div ref={bottomRef} />
        </div>

        {/* ── Input area ──────────────────────────────────────────────────── */}
        <div className="border-t border-gray-200 bg-white p-4">
          <div className="flex gap-3 max-w-4xl mx-auto">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your research papers..."
              rows={1}
              disabled={isStreaming || serverReady === false}
              className="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-3 text-sm
                         focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                         text-gray-900 disabled:bg-gray-50 disabled:text-gray-400"
            />
            <button
              onClick={sendMessage}
              disabled={isStreaming || !input.trim() || serverReady === false}
              className="bg-blue-600 text-white px-5 py-3 rounded-xl text-sm font-medium
                         hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed
                         transition-colors">
              {isStreaming ? "..." : "Send"}
            </button>
          </div>
          <p className="text-xs text-gray-400 text-center mt-2">
            Press Enter to send · Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
}