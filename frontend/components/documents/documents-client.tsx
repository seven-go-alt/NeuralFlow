"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { DocumentList } from "@/components/documents/document-list";
import { DocumentUpload } from "@/components/documents/document-upload";
import { listDocuments } from "@/services/documents";
import { isDocumentProcessing } from "@/types/documents";
import type { DocumentItem } from "@/types/documents";

export function DocumentsClient({ initialItems }: { initialItems: DocumentItem[] }) {
  const [items, setItems] = useState(initialItems);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (items !== initialItems && !loading) {
    const currentIds = items.map((item) => item.document_id).join(",");
    const nextIds = initialItems.map((item) => item.document_id).join(",");
    if (currentIds !== nextIds) {
      setItems(initialItems);
    }
  }

  const hasProcessingItems = useMemo(
    () => items.some((item) => isDocumentProcessing(item.status)),
    [items],
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listDocuments();
      setItems(data.items);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh documents");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!hasProcessingItems) return;
    const timer = window.setInterval(() => {
      void refresh();
    }, 3000);
    return () => window.clearInterval(timer);
  }, [hasProcessingItems, refresh]);

  return (
    <div className="space-y-6">
      <DocumentUpload onUploaded={refresh} />
      {loading ? <div className="text-xs text-zinc-500">Refreshing documents…</div> : null}
      {error ? <div className="rounded-md border border-rose-400/20 bg-rose-400/5 px-3 py-2 text-xs text-rose-200">{error}</div> : null}
      {hasProcessingItems ? (
        <div className="text-xs text-amber-300">Documents are still being processed. This page auto-refreshes every 3 seconds until they become ready.</div>
      ) : null}
      <DocumentList items={items} onRefresh={refresh} />
    </div>
  );
}
