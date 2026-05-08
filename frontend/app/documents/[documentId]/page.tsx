import { getDocument, getDocumentChunks } from "@/services/documents";

export default async function DocumentDetailPage({ params }: { params: Promise<{ documentId: string }> }) {
  const { documentId } = await params;
  const [document, chunks] = await Promise.all([
    getDocument(documentId),
    getDocumentChunks(documentId),
  ]);

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
      <div className="mx-auto max-w-5xl space-y-6">
        <div>
          <a href="/documents" className="text-sm text-cyan-300">← Back to documents</a>
          <div className="mt-3 text-2xl font-semibold">{document.title || document.original_filename}</div>
          <div className="mt-1 text-sm text-zinc-500">{document.original_filename} · {document.file_type.toUpperCase()} · {document.status}</div>
        </div>
        <section className="grid gap-4 md:grid-cols-4">
          {[
            ["Chunks", String(document.chunk_count)],
            ["Tokens", String(document.token_count ?? 0)],
            ["Tenant", document.tenant_id],
            ["Owner", document.owner_user_id],
          ].map(([label, value]) => (
            <div key={label} className="rounded-xl border bg-zinc-950/40 p-4">
              <div className="text-xs uppercase tracking-wide text-zinc-500">{label}</div>
              <div className="mt-2 text-lg font-semibold text-zinc-100">{value}</div>
            </div>
          ))}
        </section>
        <section className="rounded-xl border bg-zinc-950/40 p-4">
          <div className="mb-4 text-sm font-medium">Chunks</div>
          <div className="space-y-3">
            {chunks.items.map((chunk) => (
              <div key={chunk.chunk_id} className="rounded-lg border bg-black/20 p-3">
                <div className="flex items-center justify-between gap-3 text-xs text-zinc-500">
                  <span>#{chunk.chunk_index}</span>
                  <span>page {chunk.page_number ?? "-"}</span>
                  <span>{chunk.token_count} tokens</span>
                </div>
                <div className="mt-2 whitespace-pre-wrap text-sm leading-6 text-zinc-200">{chunk.content}</div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
