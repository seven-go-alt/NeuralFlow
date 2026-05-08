"use client";

import { useCallback, useEffect, useState } from "react";

import { DocumentList } from "@/components/documents/document-list";
import { DocumentUpload } from "@/components/documents/document-upload";
import { listDocuments } from "@/services/documents";
import type { DocumentItem } from "@/types/documents";

export function DocumentsClient({ initialItems }: { initialItems: DocumentItem[] }) {
  const [items, setItems] = useState(initialItems);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listDocuments();
      setItems(data.items);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setItems(initialItems);
  }, [initialItems]);

  return (
    <div className="space-y-6">
      <DocumentUpload onUploaded={refresh} />
      {loading ? <div className="text-xs text-zinc-500">Refreshing documents…</div> : null}
      <DocumentList items={items} />
    </div>
  );
}
