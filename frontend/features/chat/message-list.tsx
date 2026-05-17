"use client";

import { Boxes, Brain, Database, FileStack, MemoryStick, Sparkles, TerminalSquare, Wrench } from "lucide-react";

import { useAutoScroll } from "@/hooks/use-auto-scroll";
import type { ChatMessage } from "@/types/agent";

import { MessageBubble } from "./message-bubble";

const capabilities = [
  { title: "Retrieval", icon: Database, copy: "Chunk ranking, source scores, context assembly.", tone: "text-emerald-300" },
  { title: "Tools", icon: Wrench, copy: "Function calls, MCP routing, arguments, outputs.", tone: "text-amber-300" },
  { title: "Memory", icon: MemoryStick, copy: "Working memory, vector archive, compression.", tone: "text-violet-300" },
  { title: "Streaming", icon: TerminalSquare, copy: "SSE deltas, reasoning state, token usage.", tone: "text-cyan-300" },
];

const pipeline = [
  { label: "Intent", icon: Brain, color: "border-violet-400/30 bg-violet-400/10 text-violet-200" },
  { label: "Retrieve", icon: Database, color: "border-emerald-400/30 bg-emerald-400/10 text-emerald-200" },
  { label: "Call", icon: Boxes, color: "border-amber-400/30 bg-amber-400/10 text-amber-200" },
  { label: "Stream", icon: FileStack, color: "border-cyan-400/30 bg-cyan-400/10 text-cyan-200" },
];

export function MessageList({ messages, onRetry }: { messages: ChatMessage[]; onRetry?: () => void }) {
  const ref = useAutoScroll(messages.map((message) => `${message.id}:${message.content.length}`).join("|"));

  if (messages.length === 0) {
    return (
      <div className="h-full overflow-y-auto px-5 py-6 md:px-8">
        <div className="mx-auto w-full max-w-5xl pb-6">
          <div className="grid gap-5 2xl:grid-cols-[1.1fr_0.9fr]">
            <section className="hairline-panel rounded-xl border bg-zinc-950/55 p-5 md:p-7">
              <div className="mb-5 flex items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <div className="grid h-11 w-11 place-items-center rounded-lg border border-cyan-300/30 bg-cyan-300/10">
                    <Sparkles className="h-5 w-5 text-cyan-200" />
                  </div>
                  <div>
                    <div className="text-xs font-medium uppercase tracking-[0.2em] text-cyan-200/80">Runtime cockpit</div>
                    <h1 className="mt-1 text-2xl font-semibold tracking-normal text-zinc-50">NeuralFlow Agent Console</h1>
                  </div>
                </div>
                <div className="hidden rounded-lg border border-emerald-400/20 bg-emerald-400/10 px-3 py-2 text-xs text-emerald-200 sm:block">Observable by design</div>
              </div>
              <p className="max-w-2xl text-sm leading-6 text-zinc-400">
                A production console for agent runs: inspect RAG evidence, function calls, MCP execution, memory writes, compression summaries, token flow, and streaming reasoning in one place.
              </p>
              <div className="mt-7 rounded-xl border bg-black/20 p-4">
                <div className="mb-4 flex items-center justify-between">
                  <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Execution pipeline</div>
                  <div className="h-1 w-24 rounded-full bg-zinc-800">
                    <div className="scanline h-1 w-full rounded-full" />
                  </div>
                </div>
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  {pipeline.map(({ label, icon: Icon, color }, index) => (
                    <div key={label} className="relative rounded-lg border bg-zinc-950/80 p-3">
                      {index < pipeline.length - 1 && <div className="absolute right-[-18px] top-1/2 hidden h-px w-8 bg-zinc-700 sm:block" />}
                      <div className={`mb-3 grid h-9 w-9 place-items-center rounded-md border ${color}`}>
                        <Icon className="h-4 w-4" />
                      </div>
                      <div className="text-sm font-medium text-zinc-100">{label}</div>
                      <div className="mt-1 text-[11px] text-zinc-500">phase {index + 1}</div>
                    </div>
                  ))}
                </div>
              </div>
            </section>
            <section className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-1">
              {capabilities.map(({ title, icon: Icon, copy, tone }) => (
                <div key={title} className="hairline-panel rounded-xl border bg-zinc-950/55 p-4">
                  <div className="flex items-start gap-3">
                    <div className="grid h-9 w-9 place-items-center rounded-md border bg-zinc-900/70">
                      <Icon className={`h-4 w-4 ${tone}`} />
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-zinc-100">{title}</div>
                      <div className="mt-1 text-xs leading-5 text-zinc-500">{copy}</div>
                    </div>
                  </div>
                </div>
              ))}
            </section>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div ref={ref} className="flex-1 space-y-6 overflow-y-auto px-4 py-6 md:px-8">
      <div className="stagger-children">
        {messages.map((message) => (
          <div key={message.id} className="mb-6">
            <MessageBubble message={message} onRetry={onRetry} />
          </div>
        ))}
      </div>
    </div>
  );
}
