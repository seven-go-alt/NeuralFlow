import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, BarChart3, FileText, Gauge, TrendingUp } from "lucide-react";
import { getEvalRun } from "@/services/eval";

const colorMap: Record<string, { icon: string; value: string }> = {
  emerald: { icon: "text-emerald-400", value: "text-emerald-200" },
  cyan: { icon: "text-cyan-400", value: "text-cyan-200" },
  violet: { icon: "text-violet-400", value: "text-violet-200" },
  amber: { icon: "text-amber-400", value: "text-amber-200" },
  rose: { icon: "text-rose-400", value: "text-rose-200" },
};

function MetricCard({ icon: Icon, label, value, color }: { icon: typeof BarChart3; label: string; value: string; color: string }) {
  const c = colorMap[color] ?? colorMap.emerald;
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="flex items-center gap-2 text-xs text-zinc-500">
        <Icon className={`h-3.5 w-3.5 ${c.icon}`} />
        {label}
      </div>
      <div className={`mt-1.5 text-xl font-semibold font-mono ${c.value}`}>{value}</div>
    </div>
  );
}

export default async function EvalRunPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  const run = await getEvalRun(runId).catch(() => null);
  if (!run) notFound();

  const metrics = run.metrics as Record<string, number>;
  const hitRate = ((metrics.retrieval_hit_rate ?? 0) * 100).toFixed(0);
  const citationAcc = ((metrics.citation_accuracy ?? 0) * 100).toFixed(0);
  const kwCoverage = ((metrics.keyword_coverage ?? 0) * 100).toFixed(0);
  const rel = metrics.answer_relevance !== undefined ? `${(metrics.answer_relevance * 100).toFixed(0)}%` : "—";
  const faithful = metrics.answer_faithfulness !== undefined ? `${(metrics.answer_faithfulness * 100).toFixed(0)}%` : "—";
  const complete = metrics.answer_completeness !== undefined ? `${(metrics.answer_completeness * 100).toFixed(0)}%` : "—";

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
      <div className="mx-auto max-w-5xl space-y-6">
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

        <div className="grid grid-cols-3 gap-4">
          <MetricCard icon={Gauge} label="Retrieval Hit Rate" value={`${hitRate}%`} color="emerald" />
          <MetricCard icon={FileText} label="Citation Accuracy" value={`${citationAcc}%`} color="cyan" />
          <MetricCard icon={TrendingUp} label="Keyword Coverage" value={`${kwCoverage}%`} color="violet" />
        </div>

        {metrics.answer_relevance !== undefined && (
          <div>
            <div className="mb-3 text-sm font-medium text-zinc-300">Answer Quality (LLM-as-Judge)</div>
            <div className="grid grid-cols-3 gap-4">
              <MetricCard icon={BarChart3} label="Relevance" value={rel} color="amber" />
              <MetricCard icon={BarChart3} label="Faithfulness" value={faithful} color="emerald" />
              <MetricCard icon={BarChart3} label="Completeness" value={complete} color="cyan" />
            </div>
          </div>
        )}

        {run.completed_at && (
          <div className="text-xs text-zinc-600 font-mono">
            Started: {run.started_at} &middot; Completed: {run.completed_at}
          </div>
        )}
      </div>
    </main>
  );
}
