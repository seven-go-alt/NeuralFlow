"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { MetricTrendChart, ScoreDistChart } from "@/components/eval/score-chart";
import { listEvalRuns, triggerEvalRun } from "@/services/eval";
import type { EvalRunSummary } from "@/types/eval";

export default function EvalPage() {
  const [showTrigger, setShowTrigger] = useState(false);
  const [datasetId, setDatasetId] = useState("");
  const [topK, setTopK] = useState(5);

  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["eval-runs"],
    queryFn: listEvalRuns,
    refetchInterval: 30_000,
  });

  const triggerMutation = useMutation({
    mutationFn: () => triggerEvalRun({ dataset_id: datasetId, top_k: topK }),
    onSuccess: () => {
      setShowTrigger(false);
      setDatasetId("");
      refetch();
    },
  });

  const runs = data?.runs ?? [];

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
      <div className="mx-auto max-w-5xl space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <div className="text-2xl font-semibold">RAG Evaluation</div>
            <div className="mt-1 text-sm text-zinc-500">
              LLM-as-judge answer quality scoring and pipeline metrics
            </div>
          </div>
          <Button onClick={() => setShowTrigger(true)}>Trigger Eval Run</Button>
        </div>

        {/* Trigger dialog */}
        {showTrigger && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="w-full max-w-md rounded-xl border border-zinc-700 bg-zinc-900 p-6 shadow-xl">
              <div className="mb-4 text-sm font-semibold text-zinc-200">Trigger New Eval Run</div>
              <div className="space-y-4">
                <div>
                  <label className="text-xs text-zinc-500 font-mono">Dataset ID</label>
                  <input
                    type="text"
                    value={datasetId}
                    onChange={(e) => setDatasetId(e.target.value)}
                    placeholder="rag_quality_50"
                    aria-label="Dataset ID"
                    required
                    pattern="[A-Za-z0-9_.-]+"
                    title="Use a dataset filename or ID, not a filesystem path"
                    className="mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-200 outline-none focus:border-cyan-500"
                  />
                </div>
                <div>
                  <label className="text-xs text-zinc-500 font-mono">Top-K: {topK}</label>
                  <input
                    type="range"
                    min={1}
                    max={20}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="mt-1 w-full accent-cyan-500"
                  />
                </div>
                {triggerMutation.isError && (
                  <div className="text-xs text-rose-400">
                    {(triggerMutation.error as Error).message}
                  </div>
                )}
                <div className="flex justify-end gap-2">
                  <Button variant="ghost" onClick={() => setShowTrigger(false)}>Cancel</Button>
                  <Button
                    onClick={() => triggerMutation.mutate()}
                    disabled={!datasetId.trim() || triggerMutation.isPending}
                  >
                    {triggerMutation.isPending ? "Running…" : "Run"}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Loading */}
        {isLoading && (
          <div className="grid grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-24 animate-pulse rounded-lg bg-zinc-800/50" />
            ))}
          </div>
        )}

        {/* Error */}
        {isError && (
          <div className="rounded-lg border border-rose-800/50 bg-rose-950/20 p-4 text-sm text-rose-400">
            Failed to load eval runs: {(error as Error).message}
            <Button variant="ghost" size="sm" onClick={() => refetch()} className="ml-3">Retry</Button>
          </div>
        )}

        {/* Charts */}
        {!isLoading && !isError && runs.length > 0 && (
          <>
            <MetricTrendChart runs={runs} />
            <ScoreDistChart runs={runs} />
          </>
        )}

        {/* Empty state */}
        {!isLoading && !isError && runs.length === 0 && (
          <div className="rounded-lg border border-dashed border-zinc-700 p-12 text-center text-sm text-zinc-500">
            No evaluation runs yet. Click &quot;Trigger Eval Run&quot; to start one.
          </div>
        )}

        {/* Run list */}
        {runs.length > 0 && (
          <div>
            <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-zinc-500 font-mono">
              Run History ({runs.length})
            </div>
            <div className="space-y-3">
              {runs.map((run: EvalRunSummary) => (
                <Link
                  key={run.run_id}
                  href={`/eval/${run.run_id}`}
                  className="block rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 transition-colors hover:bg-zinc-800/50"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="text-sm font-medium text-zinc-100">{run.dataset_name}</div>
                      <div className="mt-1 text-xs text-zinc-500 font-mono">
                        {run.total_cases} cases &middot; {run.average_latency_ms.toFixed(0)}ms avg latency
                      </div>
                    </div>
                    <div className="flex items-center gap-4 text-xs font-mono">
                      <span className={`rounded px-2 py-0.5 ${run.status === "failed" ? "bg-rose-950/50 text-rose-300" : run.status === "completed" ? "bg-emerald-950/50 text-emerald-300" : "bg-amber-950/50 text-amber-300"}`}>
                        {run.status} {run.status !== "completed" && run.status !== "failed" ? `${run.progress}%` : ""}
                      </span>
                      <span className="text-zinc-400">
                        Hit: <span className="text-emerald-300">{(run.retrieval_hit_rate * 100).toFixed(0)}%</span>
                      </span>
                      {run.answer_relevance !== null && (
                        <span className="text-zinc-400">
                          Rel: <span className="text-cyan-300">{(run.answer_relevance * 100).toFixed(0)}%</span>
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="mt-2 text-[11px] text-zinc-600">{run.started_at}</div>
                </Link>
              ))}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
