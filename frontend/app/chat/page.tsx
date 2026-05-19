"use client";

import { useRef, useState, useCallback } from "react";
import { ArrowUp, MessageSquare, Radio } from "lucide-react";
import { Unbounded } from "next/font/google";

import { Badge } from "@/components/ui/badge";
import { sendChatMessage } from "@/services/chat";

const unbounded = Unbounded({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [response, setResponse] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const handleSend = useCallback(async () => {
    const message = input.trim();
    if (!message || streaming) return;

    setInput("");
    setResponse("");
    setError(null);
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const newSessionId = await sendChatMessage(
        message,
        sessionId,
        {
          onToken: (token) => setResponse((prev) => prev + token),
          onDone: (sid) => {
            setSessionId(sid);
            setStreaming(false);
          },
          onError: (err) => {
            setError(err.message);
            setStreaming(false);
          },
        },
        controller.signal,
      );
      setSessionId(newSessionId);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setStreaming(false);
        return;
      }
      setError(err instanceof Error ? err.message : "An unexpected error occurred");
      setStreaming(false);
    } finally {
      abortRef.current = null;
    }
  }, [input, sessionId, streaming]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const handleNewSession = useCallback(() => {
    setSessionId(null);
    setResponse("");
    setError(null);
    setInput("");
  }, []);

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="mx-auto flex min-h-screen max-w-4xl flex-col px-6 py-8">
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1
              className={`text-3xl font-semibold tracking-tight text-zinc-50 ${unbounded.className}`}
            >
              Chat
            </h1>
            <p className="mt-1.5 text-sm text-zinc-500">
              Streaming conversation with real-time token display
            </p>
          </div>
          {sessionId && (
            <button
              onClick={handleNewSession}
              className="rounded-md border border-zinc-800 bg-zinc-950/40 px-3 py-2 text-xs text-zinc-400 transition-colors hover:bg-zinc-900 hover:text-zinc-200 font-mono"
            >
              New session
            </button>
          )}
        </div>

        {/* Session ID */}
        {sessionId && (
          <div className="mb-4 flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/30 px-4 py-2.5">
            <Radio className="h-3.5 w-3.5 shrink-0 text-cyan-400" />
            <span className="text-xs text-zinc-400 font-mono">
              Session: <span className="text-zinc-300">{sessionId}</span>
            </span>
            <Badge tone="cyan" className="ml-auto">
              active
            </Badge>
          </div>
        )}

        {/* Response Area */}
        <div className="flex-1 space-y-4">
          {!response && !error && !streaming && (
            <div className="flex flex-col items-center justify-center py-24 text-center">
              <div className="mb-4 grid h-14 w-14 place-items-center rounded-2xl border border-zinc-700 bg-zinc-900/60">
                <MessageSquare className="h-6 w-6 text-zinc-500" />
              </div>
              <p className="text-sm text-zinc-500">
                Type a message below to start a streaming conversation
              </p>
            </div>
          )}

          {response && (
            <div className="animate-fade-in-up rounded-lg border border-zinc-800 bg-zinc-900/50 p-5">
              <div className="prose prose-invert max-w-none text-sm leading-relaxed text-zinc-200">
                {response}
              </div>
            </div>
          )}

          {streaming && (
            <div className="flex items-center gap-2 text-xs text-cyan-400 font-mono">
              <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-cyan-400" />
              Streaming response...
            </div>
          )}

          {error && (
            <div className="animate-fade-in-up rounded-lg border border-rose-800/50 bg-rose-950/20 p-4">
              <div className="flex items-start gap-3">
                <div className="mt-0.5 text-sm text-rose-300">Something went wrong</div>
              </div>
              <p className="mt-1 text-xs text-rose-400/80 font-mono">{error}</p>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="mt-6 border-t border-zinc-800 pt-4">
          <div className="flex items-end gap-3">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
              rows={2}
              disabled={streaming}
              className="min-h-[52px] flex-1 resize-none rounded-lg border border-zinc-800 bg-zinc-900/60 px-4 py-3 text-sm text-zinc-100 placeholder-zinc-600 outline-none transition-colors focus:border-cyan-400/40 focus:ring-1 focus:ring-cyan-400/20 disabled:opacity-40"
            />
            <div className="flex shrink-0 gap-2">
              {streaming ? (
                <button
                  onClick={handleStop}
                  className="flex h-[52px] w-[52px] items-center justify-center rounded-lg border border-rose-800/50 bg-rose-950/20 text-rose-300 transition-colors hover:bg-rose-950/40"
                >
                  <div className="h-4 w-4 rounded-sm bg-rose-400" />
                </button>
              ) : (
                <button
                  onClick={handleSend}
                  disabled={!input.trim()}
                  className="flex h-[52px] w-[52px] items-center justify-center rounded-lg border border-cyan-400/30 bg-cyan-400/10 text-cyan-200 transition-colors hover:bg-cyan-400/20 disabled:cursor-not-allowed disabled:opacity-30"
                >
                  <ArrowUp className="h-5 w-5" />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
