"use client";

import { FileText, LoaderCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { DocumentItem } from "@/types/documents";

export function DocumentStatusBadge({ status }: { status: string }) {
  const tone = status === "ready" ? "emerald" : status === "failed" ? "rose" : "amber";
  return <Badge tone={tone as "emerald" | "rose" | "amber"}>{status}</Badge>;
}

export function DocumentList({ items }: { items: DocumentItem[] }) {
  return (
    <div className="space-y-3">
      {items.map((item) => (
        <a key={item.document_id} href={`/documents/${item.document_id}`} className="block rounded-xl border bg-zinc-950/40 p-4 hover:bg-zinc-900/60">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex items-center gap-2 text-sm font-medium text-zinc-100">
                <FileText className="h-4 w-4 text-cyan-300" />
                <span className="truncate">{item.title || item.original_filename}</span>
              </div>
              <div className="mt-1 text-xs text-zinc-500">{item.original_filename} · {item.file_type.toUpperCase()} · {item.chunk_count} chunks</div>
            </div>
            <DocumentStatusBadge status={item.status} />
          </div>
          {item.error_message && <div className="mt-3 text-xs text-rose-300">{item.error_message}</div>}
        </a>
      ))}
      {items.length === 0 && (
        <div className="rounded-xl border border-dashed p-6 text-sm text-zinc-500">
          <LoaderCircle className="mb-2 h-4 w-4" />
          No documents yet.
        </div>
      )}
    </div>
  );
}
