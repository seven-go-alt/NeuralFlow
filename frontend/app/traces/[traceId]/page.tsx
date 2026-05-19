import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, Clock, Cpu, GitBranch } from "lucide-react";
import { getTrace } from "@/services/traces";

interface SpanNode {
  name: string;
  span_id: string;
  parent_id: string | null;
  duration_ms: number;
  start_time: string;
  end_time: string;
  metadata?: Record<string, unknown>;
  children?: SpanNode[];
}

function flattenSpans(tree: SpanNode): SpanNode[] {
  const result: SpanNode[] = [];
  function walk(node: SpanNode) {
    result.push(node);
    for (const child of node.children ?? []) walk(child);
  }
  walk(tree);
  return result;
}

function SpanRow({ span, maxDuration }: { span: SpanNode; maxDuration: number }) {
  const pct = maxDuration > 0 ? (span.duration_ms / maxDuration) * 100 : 0;
  const barColor =
    span.duration_ms < 500
      ? "bg-emerald-500/60"
      : span.duration_ms < 2000
        ? "bg-amber-500/60"
        : "bg-rose-500/60";

  return (
    <div className="flex items-center gap-3 py-2.5">
      <div className="w-48 shrink-0 text-xs font-mono text-zinc-300 truncate" title={span.name}>
        {span.name}
      </div>
      <div className="flex-1">
        <div className="h-4 w-full rounded-full bg-zinc-800/50 overflow-hidden">
          <div
            className={`h-full rounded-full ${barColor} transition-all`}
            style={{ width: `${Math.max(pct, 2)}%` }}
          />
        </div>
      </div>
      <div className="w-20 shrink-0 text-right text-xs font-mono text-zinc-400">
        {span.duration_ms.toFixed(0)}ms
      </div>
    </div>
  );
}

export default async function TraceDetailPage({ params }: { params: Promise<{ traceId: string }> }) {
  const { traceId } = await params;
  const trace = await getTrace(traceId).catch(() => null);
  if (!trace) notFound();

  const spanTree = trace.span_tree as unknown as SpanNode;
  const spans = spanTree ? flattenSpans(spanTree) : [];
  const maxDuration = spans.reduce((max, s) => Math.max(max, s.duration_ms), 0);

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
      <div className="mx-auto max-w-5xl space-y-6">
        <div className="flex items-center gap-3">
          <Link href="/traces" className="rounded-md border border-zinc-800 p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800">
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <div>
            <div className="text-2xl font-semibold">Trace Detail</div>
            <div className="mt-1 text-xs text-zinc-500 font-mono">{traceId.slice(0, 12)}</div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <Clock className="h-3.5 w-3.5 text-cyan-400" />
              Duration
            </div>
            <div className="mt-1.5 text-xl font-semibold font-mono text-cyan-200">
              {trace.total_duration_ms.toFixed(0)}ms
            </div>
          </div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <Cpu className="h-3.5 w-3.5 text-violet-400" />
              Tokens
            </div>
            <div className="mt-1.5 text-xl font-semibold font-mono text-violet-200">
              {trace.token_count ?? "—"}
            </div>
          </div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <GitBranch className="h-3.5 w-3.5 text-emerald-400" />
              Spans
            </div>
            <div className="mt-1.5 text-xl font-semibold font-mono text-emerald-200">
              {spans.length}
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-5">
          <div className="text-sm font-medium text-zinc-300">Query</div>
          <div className="mt-2 text-sm text-zinc-400 leading-relaxed">{trace.query}</div>
        </div>

        {trace.answer && (
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-5">
            <div className="text-sm font-medium text-zinc-300">Answer</div>
            <div className="mt-2 text-sm text-zinc-400 leading-relaxed whitespace-pre-wrap">{trace.answer}</div>
          </div>
        )}

        {spans.length > 0 && (
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-5">
            <div className="mb-4 text-sm font-medium text-zinc-300">Span Waterfall</div>
            <div className="flex items-center gap-3 pb-2 text-[11px] text-zinc-600 border-b border-zinc-800">
              <div className="w-48">Operation</div>
              <div className="flex-1">Timeline</div>
              <div className="w-20 text-right">Duration</div>
            </div>
            <div className="divide-y divide-zinc-800/50">
              {spans.map((span) => (
                <SpanRow key={span.span_id} span={span} maxDuration={maxDuration} />
              ))}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
