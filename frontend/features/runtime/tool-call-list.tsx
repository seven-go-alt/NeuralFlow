"use client";

import { Wrench } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { formatLatency } from "@/lib/utils";
import type { ToolCall } from "@/types/agent";

export function ToolCallList({ toolCalls }: { toolCalls: ToolCall[] }) {
  return (
    <section>
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Function Calls / MCP</h3>
        <Badge tone="cyan">{toolCalls.length}</Badge>
      </div>
      <div className="space-y-2">
        {toolCalls.length === 0 && (
          <div className="rounded-lg border p-4 text-xs leading-5 text-zinc-500">
            <Wrench className="mb-2 h-4 w-4" />
            Tool arguments and MCP outputs will be attached here.
          </div>
        )}
        {toolCalls.map((call) => (
          <div key={call.id} className="rounded-lg border bg-zinc-950/50 p-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-zinc-100">{call.name}</span>
              <Badge tone={call.status === "error" ? "rose" : call.status === "success" ? "emerald" : "amber"}>{call.status}</Badge>
            </div>
            <div className="mt-1 text-[11px] text-zinc-500">{formatLatency(call.latencyMs)}</div>
            {call.input !== undefined && <pre className="mt-2 max-h-28 overflow-auto rounded-md bg-black/25 p-2 text-[11px] text-zinc-400">{JSON.stringify(call.input, null, 2)}</pre>}
          </div>
        ))}
      </div>
    </section>
  );
}
