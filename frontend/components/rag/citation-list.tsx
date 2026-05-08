"use client";

import { FileSearch, Link2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";

export interface CitationItem {
  index: number;
  label: string;
  document_id: string;
  chunk_id: string;
  page_number?: number | null;
}

export function CitationList({ citations }: { citations: CitationItem[] }) {
  if (citations.length === 0) return null;

  return (
    <div className="mt-3 rounded-lg border border-emerald-400/20 bg-emerald-400/5 p-3">
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-emerald-200/90">
        <FileSearch className="h-3.5 w-3.5" />
        Sources
      </div>
      <div className="space-y-2">
        {citations.map((item) => (
          <a
            key={`${item.document_id}:${item.chunk_id}:${item.index}`}
            href={`/documents/${item.document_id}`}
            className="flex items-center justify-between gap-3 rounded-md border border-white/5 bg-black/20 px-3 py-2 text-xs text-zinc-300 hover:bg-black/30"
          >
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <Badge tone="emerald">[{item.index}]</Badge>
                <span className="truncate">{item.label}</span>
              </div>
              <div className="mt-1 text-[11px] text-zinc-500">
                {item.page_number ? `page ${item.page_number}` : "document source"}
              </div>
            </div>
            <Link2 className="h-3.5 w-3.5 shrink-0 text-cyan-300" />
          </a>
        ))}
      </div>
    </div>
  );
}
