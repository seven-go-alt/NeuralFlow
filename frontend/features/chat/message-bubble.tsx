"use client";

import { Bot, Clock3, Copy, RefreshCw, RotateCcw, User } from "lucide-react";

import { CitationList } from "@/components/rag/citation-list";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { compactNumber, formatLatency } from "@/lib/utils";
import type { ChatMessage } from "@/types/agent";

import { MarkdownRenderer } from "./markdown-renderer";

export function MessageBubble({ message, onRetry }: { message: ChatMessage; onRetry?: () => void }) {
  const isUser = message.role === "user";

  return (
    <article className={`group flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        className={`mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-md border ${
          isUser ? "border-violet-400/30 bg-violet-400/10" : "border-cyan-400/30 bg-cyan-400/10"
        }`}
      >
        {isUser ? <User className="h-4 w-4 text-violet-200" /> : <Bot className="h-4 w-4 text-cyan-200" />}
      </div>
      <div className={`min-w-0 max-w-[min(780px,88%)] ${isUser ? "items-end" : "items-start"} flex flex-col`}>
        <div
          className={`rounded-lg border px-4 py-3 ${
            isUser ? "border-violet-400/20 bg-violet-950/30" : "border-zinc-800 bg-zinc-950/70"
          }`}
        >
          {message.content ? (
            <MarkdownRenderer content={message.content} />
          ) : (
            <div className="flex items-center gap-2 text-sm text-zinc-400">
              <RefreshCw className="h-3.5 w-3.5 animate-spin text-cyan-300" />
              Waiting for runtime output
            </div>
          )}
          {!isUser && message.citations?.length ? <CitationList citations={message.citations} /> : null}
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-zinc-500">
          {message.intent && <Badge tone="violet">{message.intent}</Badge>}
          {message.usedSkills?.map((skill) => (
            <Badge key={skill} tone="emerald">
              {skill}
            </Badge>
          ))}
          <span className="inline-flex items-center gap-1">
            <Clock3 className="h-3 w-3" />
            {formatLatency(message.latencyMs)}
          </span>
          <span>{compactNumber(message.tokens)} tokens</span>
          {!isUser && (
            <div className="ml-1 flex opacity-0 transition-opacity group-hover:opacity-100">
              <Button variant="ghost" size="icon" title="Copy message" onClick={() => navigator.clipboard.writeText(message.content)}>
                <Copy className="h-3.5 w-3.5" />
              </Button>
              <Button variant="ghost" size="icon" title="Regenerate" onClick={onRetry}>
                <RotateCcw className="h-3.5 w-3.5" />
              </Button>
            </div>
          )}
        </div>
      </div>
    </article>
  );
}
