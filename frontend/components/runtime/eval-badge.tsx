"use client";

import { Badge } from "@/components/ui/badge";
import type { EvalScore } from "@/types/agent";

export function EvalBadge({ evalResult }: { evalResult: EvalScore }) {
  const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
  const tone = evalResult.overall >= 0.5 ? "emerald" : "amber";

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/55 p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500 font-mono">
          Answer Quality
        </span>
        <Badge tone={tone}>{pct(evalResult.overall)}</Badge>
      </div>
      <div className="grid grid-cols-3 gap-2 text-[10px] text-zinc-500 font-mono">
        {(["relevance", "faithfulness", "completeness"] as const).map((dim) => (
          <div key={dim} className="text-center">
            <div className="text-xs font-semibold text-zinc-300">
              {pct(evalResult[dim])}
            </div>
            <div>{dim.slice(0, 3)}</div>
          </div>
        ))}
      </div>
      {evalResult.reason && (
        <div className="text-[10px] text-zinc-600 truncate" title={evalResult.reason}>
          {evalResult.reason}
        </div>
      )}
    </div>
  );
}
