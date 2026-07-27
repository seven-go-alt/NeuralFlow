"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertCircle,
  BarChart3,
  Clock,
  Cpu,
  GitBranch,
  Search,
} from "lucide-react";
import { listTraces } from "@/services/traces";
import { listEvalRuns } from "@/services/eval";
import type { TraceSummary } from "@/types/traces";
import type { EvalRunSummary } from "@/types/eval";

function durationColor(ms: number): string {
  if (ms < 1000) return "text-emerald-300";
  if (ms < 5000) return "text-amber-300";
  return "text-rose-300";
}

function MetricCard({
  icon,
  label,
  value,
  sub,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: string;
  color: string;
}) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="flex items-center gap-2 text-xs text-zinc-500">
        {icon}
        {label}
      </div>
      <div className={`mt-1.5 text-xl font-semibold font-mono ${color}`}>{value}</div>
      {sub && <div className="mt-0.5 text-[11px] text-zinc-600">{sub}</div>}
    </div>
  );
}

function TraceCard({ trace }: { trace: TraceSummary }) {
  return (
    <Link
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
        <div
          className={`shrink-0 text-right text-xs font-mono ${durationColor(trace.total_duration_ms)}`}
        >
          {trace.total_duration_ms.toFixed(0)}ms
        </div>
      </div>
      <div className="mt-2 text-[11px] text-zinc-600">{trace.created_at}</div>
    </Link>
  );
}

export default function ObservabilityPage() {
  const [traces, setTraces] = useState<TraceSummary[]>([]);
  const [evalRuns, setEvalRuns] = useState<EvalRunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  useEffect(() => {
    async function load() {
      try {
        const [t, e] = await Promise.all([listTraces(), listEvalRuns()]);
        setTraces(t.traces);
        setEvalRuns(e.runs);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load data");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const filteredTraces = useMemo(
    () =>
      traces.filter((t) =>
        t.query.toLowerCase().includes(search.toLowerCase()),
      ),
    [traces, search],
  );

  const stats = useMemo(() => {
    const count = traces.length;
    if (count === 0) return { avgLatency: 0, totalTokens: 0, evalCount: 0 };
    const avgLatency =
      traces.reduce((s, t) => s + t.total_duration_ms, 0) / count;
    const totalTokens = traces.reduce(
      (s, t) => s + (t.token_count ?? 0),
      0,
    );
    return { avgLatency, totalTokens, evalCount: evalRuns.length };
  }, [traces, evalRuns]);

  if (loading) {
    return (
      <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
        <div className="mx-auto max-w-5xl space-y-6">
          <div className="animate-pulse space-y-4">
            <div className="h-8 w-48 rounded bg-zinc-800" />
            <div className="grid grid-cols-4 gap-4">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="h-24 rounded-lg bg-zinc-900" />
              ))}
            </div>
            <div className="h-64 rounded-lg bg-zinc-900" />
          </div>
        </div>
      </main>
    );
  }

  if (error) {
    return (
      <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
        <div className="mx-auto max-w-3xl text-center">
          <AlertCircle className="mx-auto h-12 w-12 text-rose-400" />
          <p className="mt-4 text-zinc-400">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 rounded-md border border-zinc-700 px-4 py-2 text-sm text-zinc-300 transition-colors hover:bg-zinc-800"
          >
            Retry
          </button>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
      <div className="mx-auto max-w-5xl space-y-6">
        <div>
          <div className="text-2xl font-semibold">Observability</div>
          <div className="mt-1 text-sm text-zinc-500">
            Pipeline traces, eval scores, and system metrics
          </div>
        </div>

        <div className="grid grid-cols-4 gap-4">
          <MetricCard
            icon={<GitBranch className="h-3.5 w-3.5 text-cyan-400" />}
            label="Total Traces"
            value={String(traces.length)}
            sub="Pipeline executions"
            color="text-cyan-200"
          />
          <MetricCard
            icon={<Clock className="h-3.5 w-3.5 text-violet-400" />}
            label="Avg Latency"
            value={`${stats.avgLatency.toFixed(0)}ms`}
            sub="Across all traces"
            color="text-violet-200"
          />
          <MetricCard
            icon={<Cpu className="h-3.5 w-3.5 text-emerald-400" />}
            label="Total Tokens"
            value={String(stats.totalTokens)}
            sub="LLM token usage"
            color="text-emerald-200"
          />
          <MetricCard
            icon={<BarChart3 className="h-3.5 w-3.5 text-amber-400" />}
            label="Eval Runs"
            value={String(stats.evalCount)}
            sub="Answer quality assessments"
            color="text-amber-200"
          />
        </div>

        <div className="flex items-center gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
            <input
              type="text"
              placeholder="Search traces by query..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full rounded-lg border border-zinc-800 bg-zinc-900/50 py-2.5 pl-10 pr-4 text-sm text-zinc-100 placeholder-zinc-500 focus:border-cyan-400/40 focus:outline-none"
            />
          </div>
          <div className="flex gap-2">
            <Link
              href="/traces"
              className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/50 px-4 py-2.5 text-sm text-zinc-300 transition-colors hover:bg-zinc-800"
            >
              <Activity className="h-4 w-4" />
              All Traces
            </Link>
            <Link
              href="/eval"
              className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/50 px-4 py-2.5 text-sm text-zinc-300 transition-colors hover:bg-zinc-800"
            >
              <BarChart3 className="h-4 w-4" />
              Eval Dashboard
            </Link>
          </div>
        </div>

        {filteredTraces.length === 0 ? (
          <div className="rounded-lg border border-dashed border-zinc-700 p-12 text-center text-sm text-zinc-500">
            {search
              ? "No traces match your search."
              : "No traces recorded yet. Send a chat message with RAG enabled to generate a trace."}
          </div>
        ) : (
          <div className="space-y-3">
            <div className="text-sm text-zinc-400 font-mono">
              Showing {filteredTraces.length} of {traces.length} traces
            </div>
            {filteredTraces.map((trace) => (
              <TraceCard key={trace.trace_id} trace={trace} />
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
