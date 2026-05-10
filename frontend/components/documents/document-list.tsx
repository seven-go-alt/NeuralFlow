"use client";

import Link from "next/link";
import { FileText, LoaderCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { DocumentActions } from "@/components/documents/document-actions";
import type { DocumentItem } from "@/types/documents";

export function DocumentStatusBadge({ status }: { status: string }) {
  const tone = status === "ready" ? "emerald" : status === "failed" ? "rose" : "amber";
  return <Badge tone={tone as "emerald" | "rose" | "amber"}>{status}</Badge>;
}

export function DocumentList({ items, onRefresh }: { items: DocumentItem[]; onRefresh?: () => void | Promise<void> }) {
  return (
    <div className="space-y-3">
      {items.map((item) => (
        <div key={item.document_id} className="rounded-xl border bg-zinc-950/40 p-4 hover:bg-zinc-900/60">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <Link href={`/documents/${item.document_id}`} className="flex items-center gap-2 text-sm font-medium text-zinc-100 hover:text-cyan-200">
                <FileText className="h-4 w-4 text-cyan-300" />
                <span className="truncate">{item.title || item.original_filename}</span>
              </Link>
              <div className="mt-1 text-xs text-zinc-500">{item.original_filename} · {item.file_type.toUpperCase()} · {item.chunk_count} chunks</div>
            </div>
            <DocumentStatusBadge status={item.status} />
          </div>
          <div className="mt-4">
            <DocumentActions document={item} compact onActionComplete={onRefresh} />
          </div>
          {item.error_message && <div className="mt-3 text-xs text-rose-300">{item.error_message}</div>}
        </div>
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
