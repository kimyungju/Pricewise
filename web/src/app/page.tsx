"use client";

import { useEffect, useRef } from "react";
import { useChatStream } from "../hooks/useChatStream";
import { ChatMessage } from "../components/ChatMessage";
import { ChatInput } from "../components/ChatInput";
import { ThemeToggle } from "../components/ThemeToggle";

const SUGGESTIONS = [
  {
    category: "Audio",
    query: "Find wireless headphones under $100",
    description: "Discover top-rated wireless audio",
  },
  {
    category: "Computing",
    query: "Compare MacBook Air prices",
    description: "Find the best deal on Apple laptops",
  },
  {
    category: "Peripherals",
    query: "Best rated mechanical keyboards",
    description: "Explore enthusiast-grade keyboards",
  },
];

export default function Home() {
  const { messages, status, sendMessage, approveToolCall, clearSession } =
    useChatStream();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const isDisabled = status === "streaming" || status === "awaiting_approval";

  return (
    <div
      className="relative z-10 flex h-screen flex-col"
      style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}
    >
      {/* Header */}
      <header
        className="flex items-center justify-between px-8 py-4"
        style={{
          background: "var(--bg-sidebar)",
          borderBottom: "3px solid var(--accent)",
        }}
      >
        <button
          onClick={clearSession}
          className="flex items-center gap-4 transition-opacity duration-150 hover:opacity-80"
        >
          <div
            className="flex h-9 w-9 items-center justify-center rounded-lg text-sm font-bold"
            style={{
              background: "var(--accent)",
              color: "var(--text-inverse)",
              fontFamily: "'Playfair Display', Georgia, serif",
            }}
          >
            a.
          </div>
          <div className="text-left">
            <h1
              className="text-xl font-semibold tracking-tight"
              style={{
                fontFamily: "'Playfair Display', Georgia, serif",
                color: "var(--text-inverse)",
              }}
            >
              aigent
            </h1>
            <p
              className="text-[10px] font-medium uppercase tracking-[0.2em]"
              style={{ color: "var(--text-inverse-muted)" }}
            >
              Your shopping companion
            </p>
          </div>
        </button>
        <div className="flex items-center gap-2">
          {messages.length > 0 && (
            <button
              onClick={clearSession}
              className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[11px] font-medium transition-all duration-150 hover:scale-[1.02] active:scale-[0.98]"
              style={{
                background: "rgba(255,255,255,0.1)",
                color: "var(--text-inverse-muted)",
                border: "1px solid rgba(255,255,255,0.1)",
              }}
            >
              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
              </svg>
              New chat
            </button>
          )}
          <ThemeToggle />
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-6 py-6">
          {messages.length === 0 && (
            <div className="flex h-[60vh] flex-col items-center justify-center text-center">
              <div
                className="mb-4 h-px w-12"
                style={{ background: "var(--accent)" }}
              />
              <h2
                className="mb-3 text-3xl tracking-tight"
                style={{
                  fontFamily: "'Playfair Display', Georgia, serif",
                  color: "var(--text-primary)",
                }}
              >
                What are you{" "}
                <em style={{ color: "var(--accent)" }}>looking</em> for?
              </h2>
              <p
                className="max-w-sm text-sm leading-relaxed"
                style={{ color: "var(--text-muted)" }}
              >
                Search for products, compare prices across retailers, and read
                reviews — all in one place.
              </p>
              <div className="mt-10 flex flex-col gap-2 sm:flex-row sm:gap-3">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s.query}
                    onClick={() => sendMessage(s.query)}
                    className="group flex flex-col items-start rounded-xl px-5 py-4 text-left transition-all duration-200 hover:scale-[1.02]"
                    style={{
                      background: "var(--bg-elevated)",
                      border: "1px solid var(--border-light)",
                      boxShadow: "var(--shadow-sm)",
                    }}
                  >
                    <span
                      className="text-[10px] font-semibold uppercase tracking-[0.15em]"
                      style={{ color: "var(--accent)" }}
                    >
                      {s.category}
                    </span>
                    <span
                      className="mt-1 text-sm font-medium"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {s.description}
                    </span>
                    <span
                      className="mt-1.5 flex items-center gap-1 text-xs transition-transform duration-200 group-hover:translate-x-0.5"
                      style={{ color: "var(--text-muted)" }}
                    >
                      Try it
                      <svg
                        className="h-3 w-3"
                        fill="none"
                        viewBox="0 0 24 24"
                        strokeWidth={2}
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3"
                        />
                      </svg>
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="space-y-4">
            {messages.map((msg, i) => (
              <ChatMessage
                key={msg.id}
                message={msg}
                staggerIndex={i}
                onApprove={() => approveToolCall(true)}
                onDeny={() => approveToolCall(false)}
              />
            ))}
          </div>

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input area */}
      <div
        className="relative"
        style={{
          borderTop: "1px solid var(--border)",
          background: "var(--bg-primary)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
        }}
      >
        {status === "streaming" && (
          <div
            className="absolute -top-9 left-1/2 -translate-x-1/2 flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-medium animate-fade-in-up"
            style={{
              background: "var(--bg-sidebar)",
              color: "var(--text-inverse)",
              boxShadow: "var(--shadow-md)",
            }}
          >
            <span className="flex items-center gap-1">
              <span
                className="typing-dot inline-block h-1.5 w-1.5 rounded-full"
                style={{ background: "var(--accent)" }}
              />
              <span
                className="typing-dot inline-block h-1.5 w-1.5 rounded-full"
                style={{ background: "var(--accent)" }}
              />
              <span
                className="typing-dot inline-block h-1.5 w-1.5 rounded-full"
                style={{ background: "var(--accent)" }}
              />
            </span>
            Searching
          </div>
        )}
        {status === "awaiting_approval" && (
          <div
            className="absolute -top-9 left-1/2 -translate-x-1/2 rounded-full px-4 py-1.5 text-xs font-medium animate-fade-in-up"
            style={{
              background: "#fef3c7",
              color: "#92400e",
              boxShadow: "var(--shadow-sm)",
            }}
          >
            Approval needed
          </div>
        )}
        <div className="mx-auto max-w-3xl px-6 py-4">
          <ChatInput onSend={sendMessage} disabled={isDisabled} />
        </div>
      </div>
    </div>
  );
}
