"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { CaseResult } from "@/types/eval";

function pct(v: number | null | undefined): string {
  return v != null ? `${(v * 100).toFixed(0)}%` : "—";
}

function CaseRow({ item }: { item: CaseResult }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-3 p-3 text-left transition-colors hover:bg-zinc-800/30"
      >
        {open ? <ChevronDown className="h-4 w-4 shrink-0 text-zinc-500" /> : <ChevronRight className="h-4 w-4 shrink-0 text-zinc-500" />}
        <span className="flex-1 truncate text-sm text-zinc-200">{item.question}</span>
        <Badge tone={item.retrieval_hit ? "emerald" : "rose"}>{item.retrieval_hit ? "Hit" : "Miss"}</Badge>
        {item.answer_relevance != null && (
          <span className="text-xs font-mono text-zinc-400">{pct(item.answer_relevance)}</span>
        )}
      </button>
      {open && (
        <div className="space-y-2 border-t border-zinc-800 p-3 text-xs text-zinc-400 font-mono">
          <div><span className="text-zinc-500">Question:</span> {item.question}</div>
          <div><span className="text-zinc-500">Answer:</span> {item.answer ?? "(no answer)"}</div>
          <div className="flex flex-wrap gap-3">
            <span>Hit: <span className={item.retrieval_hit ? "text-emerald-300" : "text-rose-300"}>{String(item.retrieval_hit)}</span></span>
            <span>Citation: <span className={item.citation_match ? "text-emerald-300" : "text-rose-300"}>{String(item.citation_match)}</span></span>
            <span>KwCov: {pct(item.keyword_coverage)}</span>
            <span>Rel: {pct(item.answer_relevance)}</span>
            <span>Faith: {pct(item.answer_faithfulness)}</span>
            <span>Compl: {pct(item.answer_completeness)}</span>
          </div>
          <div>Latency: {item.latency_ms.toFixed(0)}ms &middot; Rank: {item.first_relevant_rank} &middot; P@{item.precision_at_k.toFixed(2)} &middot; R@{item.recall_at_k.toFixed(2)}</div>
          {item.retrieved_doc_ids.length > 0 && (
            <div>
              <span className="text-zinc-500">Docs:</span> {item.retrieved_doc_ids.join(", ")}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function CaseList({ cases }: { cases: CaseResult[] }) {
  if (cases.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-zinc-700 p-6 text-center text-sm text-zinc-500">
        No per-case results available for this run.
      </div>
    );
  }

  return (
    <div>
      <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-zinc-500 font-mono">
        Per-Case Results ({cases.length})
      </div>
      <div className="space-y-2">
        {cases.map((item) => (
          <CaseRow key={item.case_id} item={item} />
        ))}
      </div>
    </div>
  );
}
