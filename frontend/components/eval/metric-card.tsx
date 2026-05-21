"use client";

import type { LucideIcon } from "lucide-react";

const colorMap: Record<string, { icon: string; value: string }> = {
  emerald: { icon: "text-emerald-400", value: "text-emerald-200" },
  cyan: { icon: "text-cyan-400", value: "text-cyan-200" },
  violet: { icon: "text-violet-400", value: "text-violet-200" },
  amber: { icon: "text-amber-400", value: "text-amber-200" },
  rose: { icon: "text-rose-400", value: "text-rose-200" },
};

export function MetricCard({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  color: "emerald" | "cyan" | "violet" | "amber" | "rose";
}) {
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
