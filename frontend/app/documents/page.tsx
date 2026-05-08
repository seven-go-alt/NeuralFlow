import { DocumentsClient } from "@/components/documents/documents-client";
import { listDocuments } from "@/services/documents";

export default async function DocumentsPage() {
  const data = await listDocuments().catch(() => ({ items: [], total: 0, page: 1, page_size: 20 }));

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
      <div className="mx-auto max-w-5xl space-y-6">
        <div>
          <div className="text-2xl font-semibold">Documents</div>
          <div className="mt-1 text-sm text-zinc-500">Enterprise knowledge base ingestion pipeline</div>
        </div>
        <DocumentsClient initialItems={data.items} />
      </div>
    </main>
  );
}
