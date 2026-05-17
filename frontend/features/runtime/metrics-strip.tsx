"use client";

import { Activity, Gauge, Timer, Wrench } from "lucide-react";

import { compactNumber, formatLatency } from "@/lib/utils";
import type { RuntimeMetrics } from "@/types/agent";

export function MetricsStrip({ metrics }: { metrics: RuntimeMetrics }) {
  const items = [
    { label: "Tokens in", value: compactNumber(metrics.tokensIn), icon: Activity, tone: "text-violet-300" },
    { label: "Tokens out", value: compactNumber(metrics.tokensOut), icon: Gauge, tone: "text-cyan-300" },
    { label: "Latency", value: formatLatency(metrics.latencyMs), icon: Timer, tone: "text-emerald-300" },
    { label: "Tools", value: formatLatency(metrics.toolMs), icon: Wrench, tone: "text-amber-300" },
  ];

  return (
    <div className="grid grid-cols-2 gap-2">
      {items.map(({ label, value, icon: Icon, tone }) => (
        <div key={label} className="hairline-panel rounded-lg border bg-zinc-950/55 p-3">
          <div className="flex items-center gap-2 text-[11px] text-zinc-500">
            <Icon className={`h-3.5 w-3.5 ${tone}`} />
            {label}
          </div>
          <div className="mt-1 text-sm font-semibold text-zinc-100 font-mono">{value}</div>
        </div>
      ))}
    </div>
  );
}
