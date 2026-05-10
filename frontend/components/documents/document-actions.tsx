"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { FileText, MessageSquareText, RefreshCw, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { reindexDocument } from "@/services/documents";
import { useAgentStore } from "@/store/agent-store";
import { isDocumentFailed, isDocumentProcessing, isDocumentReady } from "@/types/documents";

export function DocumentActions({
  document,
  compact = false,
  onActionComplete,
}: {
  document: { document_id: string; title?: string | null; original_filename: string; status: string; error_message?: string | null };
  compact?: boolean;
  onActionComplete?: () => void | Promise<void>;
}) {
  const router = useRouter();
  const createSessionWithDocument = useAgentStore((state) => state.createSessionWithDocument);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  const title = document.title || document.original_filename;
  const ready = isDocumentReady(document.status);
  const failed = isDocumentFailed(document.status);
  const processing = isDocumentProcessing(document.status);

  function openChat(initialPrompt: string) {
    if (!ready) return;
    createSessionWithDocument(
      {
        documentId: document.document_id,
        title,
        status: document.status,
      },
      { initialPrompt },
    );
    router.push("/");
  }

  async function onReindex() {
    setBusy(true);
    setMessage("");
    try {
      await reindexDocument(document.document_id);
      setMessage("Reindex queued. The page will refresh while the document is processed.");
      await onActionComplete?.();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Failed to reindex document");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className={compact ? "" : "rounded-xl border bg-zinc-950/40 p-4"}>
      {!compact ? (
        <div className="mb-3 flex items-center gap-2 text-sm font-medium text-zinc-100">
          <FileText className="h-4 w-4 text-cyan-300" />
          Document actions
        </div>
      ) : null}
      <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap">
        <Button type="button" size={compact ? "sm" : "default"} onClick={() => openChat(`请总结这份文档《${title}》的核心内容，并提炼 5 条最重要的信息。`)} disabled={!ready || busy}>
          <Sparkles className="h-4 w-4" />
          Summarize this document
        </Button>
        <Button type="button" size={compact ? "sm" : "default"} variant="outline" onClick={() => openChat(`基于文档《${title}》，回答我的问题，并在回答里尽量引用文档内容。`)} disabled={!ready || busy}>
          <MessageSquareText className="h-4 w-4" />
          Chat with this document
        </Button>
        {(failed || !ready) ? (
          <Button type="button" size={compact ? "sm" : "default"} variant="secondary" onClick={onReindex} disabled={busy || processing}>
            <RefreshCw className={`h-4 w-4 ${busy ? "animate-spin" : ""}`} />
            {busy ? "Queueing..." : "Reindex"}
          </Button>
        ) : null}
      </div>
      <div className="mt-3 text-xs text-zinc-500">
        {ready
          ? compact
            ? "Ready for retrieval and scoped chat."
            : "This document is ready for retrieval. New chat sessions from here will be scoped to this document."
          : failed
            ? `This document failed to process${document.error_message ? `: ${document.error_message}` : "."} Reindex it after fixing the worker or source file.`
            : processing
              ? `This document is currently ${document.status}. Wait until it becomes ready before summarizing or asking questions.`
              : `This document is not ready yet. You can queue a reindex if processing did not start correctly.`}
      </div>
      {message ? <div className="mt-2 text-xs text-cyan-200">{message}</div> : null}
    </section>
  );
}
