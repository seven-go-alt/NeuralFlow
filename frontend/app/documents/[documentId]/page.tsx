import Link from "next/link";
import { Calendar, Clock, Database, FileText, HardDrive } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { DocumentActions } from "@/components/documents/document-actions";
import { getDocument, getDocumentChunks } from "@/services/documents";
import { isDocumentFailed, isDocumentProcessing, isDocumentReady } from "@/types/documents";

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** i).toFixed(1)} ${units[i]}`;
}

function formatDate(dateStr: string | null | undefined): string {
  if (!dateStr) return "—";
  const date = new Date(dateStr);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function statusBadgeTone(status: string): "emerald" | "amber" | "rose" | "zinc" {
  if (isDocumentReady(status)) return "emerald";
  if (isDocumentFailed(status)) return "rose";
  if (isDocumentProcessing(status)) return "amber";
  return "zinc";
}

export default async function DocumentDetailPage({
  params,
}: {
  params: Promise<{ documentId: string }>;
}) {
  const { documentId } = await params;
  const [document, chunks] = await Promise.all([
    getDocument(documentId).catch(() => null),
    getDocumentChunks(documentId).catch(() => ({ items: [], total: 0 })),
  ]);

  if (!document) {
    return (
      <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
        <div className="mx-auto max-w-5xl">
          <Link
            href="/documents"
            className="text-sm text-cyan-400 transition-colors hover:text-cyan-300"
          >
            &larr; Back to documents
          </Link>
          <div className="mt-8 rounded-lg border border-rose-800/50 bg-rose-950/20 p-6 text-center">
            <p className="text-rose-300">Document not found</p>
          </div>
        </div>
      </main>
    );
  }

  const title = document.title || document.original_filename;

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
      <div className="mx-auto max-w-5xl space-y-6">
        {/* Back link */}
        <div>
          <Link
            href="/documents"
            className="text-sm text-cyan-400 transition-colors hover:text-cyan-300"
          >
            &larr; Back to documents
          </Link>
        </div>

        {/* Header */}
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-6">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-3">
                <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg border border-zinc-700 bg-zinc-900">
                  <FileText className="h-5 w-5 text-cyan-300" />
                </div>
                <div className="min-w-0">
                  <h1 className="truncate text-xl font-semibold text-zinc-50">
                    {title}
                  </h1>
                  <p className="mt-0.5 truncate text-sm text-zinc-500">
                    {document.original_filename}
                    <span className="mx-2">&middot;</span>
                    {document.file_type.toUpperCase()}
                  </p>
                </div>
              </div>
            </div>
            <Badge
              tone={statusBadgeTone(document.status)}
              pulse={isDocumentReady(document.status)}
            >
              {document.status}
            </Badge>
          </div>
          {document.error_message && (
            <div className="mt-4 rounded-md border border-rose-800/30 bg-rose-950/15 p-3 text-xs text-rose-300">
              {document.error_message}
            </div>
          )}
          {document.failed_stage && (
            <div className="mt-2 rounded-md border border-amber-800/30 bg-amber-950/15 p-3 text-xs text-amber-300">
              Failed at stage: {document.failed_stage}
            </div>
          )}
        </div>

        {/* Metadata Grid */}
        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
            <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-zinc-500">
              <HardDrive className="h-3.5 w-3.5" />
              Size
            </div>
            <div className="mt-2 text-lg font-semibold text-zinc-100">
              {formatBytes(document.size_bytes)}
            </div>
          </div>
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
            <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-zinc-500">
              <Database className="h-3.5 w-3.5" />
              Chunks
            </div>
            <div className="mt-2 text-lg font-semibold text-zinc-100">
              {document.chunk_count}
            </div>
            {document.token_count != null && (
              <div className="mt-1 text-xs text-zinc-600">
                {document.token_count.toLocaleString()} tokens
              </div>
            )}
          </div>
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
            <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-zinc-500">
              <Calendar className="h-3.5 w-3.5" />
              Created
            </div>
            <div className="mt-2 text-sm font-medium text-zinc-100">
              {formatDate(document.created_at)}
            </div>
          </div>
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
            <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-zinc-500">
              <Clock className="h-3.5 w-3.5" />
              Updated
            </div>
            <div className="mt-2 text-sm font-medium text-zinc-100">
              {formatDate(document.updated_at)}
            </div>
            {document.indexed_at && (
              <div className="mt-1 text-xs text-zinc-600">
                Indexed: {formatDate(document.indexed_at)}
              </div>
            )}
          </div>
        </section>

        {/* Additional Details */}
        <section className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
          <div className="mb-3 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-zinc-500">
            <FileText className="h-3.5 w-3.5 text-zinc-400" />
            Details
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <div>
              <div className="text-xs text-zinc-600">File Type</div>
              <div className="mt-0.5 text-sm text-zinc-200 font-mono">
                {document.file_type.toUpperCase()}
              </div>
            </div>
            <div>
              <div className="text-xs text-zinc-600">MIME Type</div>
              <div className="mt-0.5 truncate text-sm text-zinc-200 font-mono">
                {document.mime_type}
              </div>
            </div>
            <div>
              <div className="text-xs text-zinc-600">Tenant ID</div>
              <div className="mt-0.5 truncate text-sm text-zinc-200 font-mono">
                {document.tenant_id}
              </div>
            </div>
            <div>
              <div className="text-xs text-zinc-600">Owner</div>
              <div className="mt-0.5 truncate text-sm text-zinc-200 font-mono">
                {document.owner_user_id}
              </div>
            </div>
            <div>
              <div className="text-xs text-zinc-600">Document ID</div>
              <div className="mt-0.5 truncate text-sm text-zinc-200 font-mono">
                {document.document_id}
              </div>
            </div>
            <div>
              <div className="text-xs text-zinc-600">Checksum (SHA-256)</div>
              <div
                className="mt-0.5 truncate text-sm text-zinc-200 font-mono"
                title={document.checksum_sha256}
              >
                {document.checksum_sha256.substring(0, 16)}...
              </div>
            </div>
          </div>
        </section>

        {/* Document Actions */}
        <DocumentActions document={document} />

        {/* Chunks */}
        {chunks.items.length > 0 && (
          <section className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
            <div className="mb-4 flex items-center gap-2 text-sm font-medium text-zinc-100">
              <Database className="h-4 w-4 text-cyan-300" />
              Chunks ({chunks.total})
            </div>
            <div className="space-y-3">
              {chunks.items.map((chunk) => (
                <div
                  key={chunk.chunk_id}
                  className="rounded-lg border border-zinc-800 bg-black/30 p-3"
                >
                  <div className="flex items-center justify-between gap-3 text-xs text-zinc-500">
                    <span className="font-mono">#{chunk.chunk_index}</span>
                    <span>
                      Page {chunk.page_number ?? "—"}
                    </span>
                    <span>{chunk.token_count} tokens</span>
                  </div>
                  <div className="mt-2 whitespace-pre-wrap text-sm leading-6 text-zinc-300">
                    {chunk.content}
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
