import * as React from "react";

import { cn } from "@/lib/utils";

type BadgeTone = "cyan" | "emerald" | "amber" | "rose" | "violet" | "zinc";

const tones: Record<BadgeTone, string> = {
  cyan: "border-cyan-400/30 bg-cyan-400/10 text-cyan-200",
  emerald: "border-emerald-400/30 bg-emerald-400/10 text-emerald-200",
  amber: "border-amber-400/30 bg-amber-400/10 text-amber-200",
  rose: "border-rose-400/30 bg-rose-400/10 text-rose-200",
  violet: "border-violet-400/30 bg-violet-400/10 text-violet-200",
  zinc: "border-zinc-700 bg-zinc-800 text-zinc-300",
};

export function Badge({
  className,
  tone = "zinc",
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { tone?: BadgeTone }) {
  return (
    <span
      className={cn("inline-flex items-center rounded-md border px-2 py-0.5 text-[11px] font-medium", tones[tone], className)}
      {...props}
    />
  );
}
