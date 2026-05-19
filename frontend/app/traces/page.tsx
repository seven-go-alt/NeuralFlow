import Link from "next/link";
import { listTraces } from "@/services/traces";

const durationColorClass: Record<string, string> = {
  fast: "text-emerald-300",
  medium: "text-amber-300",
  slow: "text-rose-300",
};

function durationColor(ms: number): string {
  if (ms < 1000) return durationColorClass.fast;
  if (ms < 5000) return durationColorClass.medium;
  return durationColorClass.slow;
}

export default async function TracesPage() {
  const data = await listTraces().catch(() => ({ traces: [] }));

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
      <div className="mx-auto max-w-5xl space-y-6">
        <div>
          <div className="text-2xl font-semibold">RAG Traces</div>
          <div className="mt-1 text-sm text-zinc-500">
            Pipeline execution traces with per-stage timing
          </div>
        </div>

        {data.traces.length === 0 ? (
          <div className="rounded-lg border border-dashed border-zinc-700 p-12 text-center text-sm text-zinc-500">
            No traces recorded yet. Enable RAG and send a chat message to generate a trace.
          </div>
        ) : (
          <div className="space-y-3">
            {data.traces.map((trace) => (
              <Link
                key={trace.trace_id}
                href={`/traces/${trace.trace_id}`}
                className="block rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 transition-colors hover:bg-zinc-800/50"
              >
                <div className="flex items-start justify-between">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-medium text-zinc-100">
                      {trace.query}
                    </div>
                    <div className="mt-1 flex items-center gap-3 text-xs text-zinc-500 font-mono">
                      <span>{trace.session_id.slice(0, 12)}</span>
                      {trace.token_count !== null && (
                        <span>{trace.token_count} tokens</span>
                      )}
                    </div>
                  </div>
                  <div className={`shrink-0 text-right text-xs font-mono ${durationColor(trace.total_duration_ms)}`}>
                    {trace.total_duration_ms.toFixed(0)}ms
                  </div>
                </div>
                <div className="mt-2 text-[11px] text-zinc-600">{trace.created_at}</div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
