"use client";

import { Database } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { RetrievedChunk } from "@/types/agent";

export function RetrievedChunks({ chunks }: { chunks: RetrievedChunk[] }) {
  return (
    <section>
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Retrieved Chunks</h3>
        <Badge tone="emerald">{chunks.length}</Badge>
      </div>
      <div className="space-y-2">
        {chunks.length === 0 && (
          <div className="rounded-lg border p-4 text-xs leading-5 text-zinc-500">
            <Database className="mb-2 h-4 w-4" />
            RAG evidence will appear after retrieval executes.
          </div>
        )}
        {chunks.map((chunk) => (
          <div key={chunk.id} className="rounded-lg border bg-zinc-950/50 p-3">
            <div className="flex items-center justify-between gap-2">
              <span className="truncate text-xs font-medium text-zinc-300">{chunk.source}</span>
              <Badge tone="emerald">{Math.round(chunk.score * 100)}%</Badge>
            </div>
            <p className="mt-2 line-clamp-4 text-xs leading-5 text-zinc-500">{chunk.text}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
