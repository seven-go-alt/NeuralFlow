"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Radar,
  RadarChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { EvalRunSummary } from "@/types/eval";

function fracToPct(v: number | null | undefined): number {
  return v != null ? +(v * 100).toFixed(1) : 0;
}

export function MetricTrendChart({ runs }: { runs: EvalRunSummary[] }) {
  const recent = runs.slice(0, 20).reverse();
  const data = recent.map((r) => ({
    name: r.dataset_name.length > 12 ? r.dataset_name.slice(0, 12) + "…" : r.dataset_name,
    "Hit Rate": fracToPct(r.retrieval_hit_rate),
    Relevance: fracToPct(r.answer_relevance),
    Faithfulness: fracToPct(r.answer_faithfulness),
  }));

  if (data.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-zinc-700 text-sm text-zinc-500">
        No runs to chart yet
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-zinc-500 font-mono">
        Metric Trend (latest {data.length})
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#71717a" }} interval="preserveStartEnd" />
          <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#71717a" }} />
          <Tooltip
            contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: "8px", fontSize: "12px" }}
          />
          <Legend wrapperStyle={{ fontSize: "10px" }} />
          <Line type="monotone" dataKey="Hit Rate" stroke="#34d399" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="Relevance" stroke="#22d3ee" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="Faithfulness" stroke="#a78bfa" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ScoreDistChart({ runs }: { runs: EvalRunSummary[] }) {
  const data = runs.slice(0, 20).reverse().map((r) => ({
    name: r.run_id.slice(0, 8),
    Rel: fracToPct(r.answer_relevance),
    Faith: fracToPct(r.answer_faithfulness),
    Compl: fracToPct(r.answer_completeness),
  }));

  if (data.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-zinc-700 text-sm text-zinc-500">
        No score data yet
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-zinc-500 font-mono">
        Answer Quality Scores
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#71717a" }} />
          <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#71717a" }} />
          <Tooltip
            contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: "8px", fontSize: "12px" }}
          />
          <Legend wrapperStyle={{ fontSize: "10px" }} />
          <Bar dataKey="Rel" fill="#f59e0b" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Faith" fill="#34d399" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Compl" fill="#22d3ee" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function AnswerRadar({
  relevance,
  faithfulness,
  completeness,
}: {
  relevance: number | null | undefined;
  faithfulness: number | null | undefined;
  completeness: number | null | undefined;
}) {
  const hasData = relevance != null || faithfulness != null || completeness != null;

  if (!hasData) {
    return (
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 text-center text-sm text-zinc-500">
        No LLM-as-judge evaluations were run for this batch.
      </div>
    );
  }

  const data = [
    { dimension: "Relevance", score: fracToPct(relevance) },
    { dimension: "Faithfulness", score: fracToPct(faithfulness) },
    { dimension: "Completeness", score: fracToPct(completeness) },
  ];

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-zinc-500 font-mono">
        Answer Quality Radar
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <RadarChart data={data}>
          <PolarGrid stroke="#27272a" />
          <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 11, fill: "#a1a1aa" }} />
          <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#71717a" }} />
          <Radar dataKey="score" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.25} strokeWidth={2} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
