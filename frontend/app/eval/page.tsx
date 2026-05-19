import Link from "next/link";
import { listEvalRuns } from "@/services/eval";

export default async function EvalPage() {
  const data = await listEvalRuns().catch(() => ({ runs: [] }));

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
      <div className="mx-auto max-w-5xl space-y-6">
        <div>
          <div className="text-2xl font-semibold">RAG Evaluation</div>
          <div className="mt-1 text-sm text-zinc-500">
            LLM-as-judge answer quality scoring and pipeline metrics
          </div>
        </div>

        {data.runs.length === 0 ? (
          <div className="rounded-lg border border-dashed border-zinc-700 p-12 text-center text-sm text-zinc-500">
            No evaluation runs yet. Trigger one via <code className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-xs text-zinc-300">POST /api/v1/eval/run</code>.
          </div>
        ) : (
          <div className="space-y-3">
            {data.runs.map((run) => (
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
        )}
      </div>
    </main>
  );
}
