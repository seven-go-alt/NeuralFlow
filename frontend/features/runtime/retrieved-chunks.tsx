"use client";

import Link from "next/link";
import { AlertTriangle, Database, ExternalLink, Info, OctagonAlert } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { useAgentStore } from "@/store/agent-store";
import type { RetrievedChunk, RuntimeHint } from "@/types/agent";

function HintCard({ hint }: { hint: RuntimeHint }) {
  const Icon = hint.kind === "error" ? OctagonAlert : hint.kind === "warning" ? AlertTriangle : Info;
  const tone = hint.kind === "error" ? "text-rose-300" : hint.kind === "warning" ? "text-amber-300" : "text-cyan-300";

  return (
    <div className="rounded-lg border p-4 text-xs leading-5 text-zinc-400">
      <Icon className={`mb-2 h-4 w-4 ${tone}`} />
      <div className="font-medium text-zinc-200">{hint.title}</div>
      <div className="mt-1">{hint.detail}</div>
    </div>
  );
}

export function RetrievedChunks({ chunks }: { chunks: RetrievedChunk[] }) {
  const hint = useAgentStore((state) => state.runtime.hint);

  return (
    <section>
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Retrieved Chunks</h3>
        <Badge tone="emerald">{chunks.length}</Badge>
      </div>
      <div className="space-y-2">
        {chunks.length === 0 && hint ? <HintCard hint={hint} /> : null}
        {chunks.length === 0 && !hint && (
          <div className="rounded-lg border p-4 text-xs leading-5 text-zinc-500">
            <Database className="mb-2 h-4 w-4" />
            RAG evidence will appear after retrieval executes.
          </div>
        )}
        {chunks.map((chunk) => (
          <div key={chunk.id} className="rounded-lg border bg-zinc-950/50 p-3 animate-fade-in-up">
            <div className="flex items-center justify-between gap-2">
              <span className="truncate text-xs font-medium text-zinc-300 font-mono">{chunk.title || chunk.source}</span>
              <Badge tone="emerald">{Math.round(chunk.score * 100)}%</Badge>
            </div>
            <div className="mt-1 flex items-center gap-2 text-[11px] text-zinc-500 font-mono">
              <span>{chunk.source}</span>
              {chunk.pageNumber ? <span>· p.{chunk.pageNumber}</span> : null}
              {chunk.documentId ? (
                <Link href={`/documents/${chunk.documentId}`} className="inline-flex items-center gap-1 text-cyan-300 hover:text-cyan-200">
                  source
                  <ExternalLink className="h-3 w-3" />
                </Link>
              ) : null}
            </div>
            <p className="mt-2 line-clamp-5 text-xs leading-5 text-zinc-500">{chunk.text}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
