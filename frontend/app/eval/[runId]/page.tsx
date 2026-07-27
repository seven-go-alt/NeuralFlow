"use client";

import { use } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, BarChart3, FileText, Gauge, TrendingUp } from "lucide-react";
import Link from "next/link";
import { notFound } from "next/navigation";

import { MetricCard } from "@/components/eval/metric-card";
import { AnswerRadar } from "@/components/eval/score-chart";
import { CaseList } from "@/components/eval/case-list";
import { getEvalRun } from "@/services/eval";

export default function EvalRunPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = use(params);
  const { data: run, isLoading, isError } = useQuery({
    queryKey: ["eval-run", runId],
    queryFn: () => getEvalRun(runId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "completed" || status === "failed" ? false : 2000;
    },
  });

  if (isLoading) {
    return (
      <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
        <div className="mx-auto max-w-5xl space-y-6">
          <div className="h-6 w-64 animate-pulse rounded bg-zinc-800" />
          <div className="grid grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => <div key={i} className="h-24 animate-pulse rounded-lg bg-zinc-800/50" />)}
          </div>
        </div>
      </main>
    );
  }

  if (isError || !run) notFound();

  if (run.status !== "completed") {
    return (
      <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
        <div className="mx-auto max-w-5xl space-y-6">
          <Link href="/eval" className="text-sm text-zinc-400 hover:text-zinc-200">← Back to evaluations</Link>
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-6">
            <div className="text-lg font-semibold">Evaluation {run.status}</div>
            <div className="mt-2 text-sm text-zinc-400">Progress: {run.progress}%</div>
            {run.error_message && <div className="mt-3 text-sm text-rose-300">{run.error_message}</div>}
          </div>
        </div>
      </main>
    );
  }

  const m = run.metrics;

  const hitRate = ((m.retrieval_hit_rate ?? 0) * 100).toFixed(0);
  const citationAcc = ((m.citation_accuracy ?? 0) * 100).toFixed(0);
  const kwCoverage = ((m.keyword_coverage ?? 0) * 100).toFixed(0);
  const rel = m.answer_relevance != null ? `${(m.answer_relevance * 100).toFixed(0)}%` : "—";
  const faithful = m.answer_faithfulness != null ? `${(m.answer_faithfulness * 100).toFixed(0)}%` : "—";
  const complete = m.answer_completeness != null ? `${(m.answer_completeness * 100).toFixed(0)}%` : "—";
  const hasAnswerQuality = m.answer_relevance != null || m.answer_faithfulness != null || m.answer_completeness != null;

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
      <div className="mx-auto max-w-5xl space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <Link href="/eval" className="rounded-md border border-zinc-800 p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800">
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <div>
            <div className="text-2xl font-semibold">{run.dataset_name}</div>
            <div className="mt-1 text-xs text-zinc-500 font-mono">
              {run.total_cases} cases &middot; {run.run_id.slice(0, 8)}
            </div>
          </div>
        </div>

        {/* Retrieval metric cards */}
        <div className="grid grid-cols-3 gap-4">
          <MetricCard icon={Gauge} label="Retrieval Hit Rate" value={`${hitRate}%`} color="emerald" />
          <MetricCard icon={FileText} label="Citation Accuracy" value={`${citationAcc}%`} color="cyan" />
          <MetricCard icon={TrendingUp} label="Keyword Coverage" value={`${kwCoverage}%`} color="violet" />
        </div>

        {/* Answer quality section */}
        {hasAnswerQuality && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div>
              <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-zinc-500 font-mono">
                Answer Quality (LLM-as-Judge)
              </div>
              <div className="grid grid-cols-3 gap-4">
                <MetricCard icon={BarChart3} label="Relevance" value={rel} color="amber" />
                <MetricCard icon={BarChart3} label="Faithfulness" value={faithful} color="emerald" />
                <MetricCard icon={BarChart3} label="Completeness" value={complete} color="cyan" />
              </div>
            </div>
            <AnswerRadar
              relevance={m.answer_relevance}
              faithfulness={m.answer_faithfulness}
              completeness={m.answer_completeness}
            />
          </div>
        )}

        {/* Per-case drill-down */}
        <CaseList cases={run.per_case_results?.results ?? []} />

        {/* Footer */}
        {run.completed_at && (
          <div className="text-xs text-zinc-600 font-mono">
            Started: {run.started_at} &middot; Completed: {run.completed_at}
          </div>
        )}
      </div>
    </main>
  );
}
