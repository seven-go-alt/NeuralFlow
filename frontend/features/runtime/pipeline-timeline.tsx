"use client";

import { Brain, CheckCircle2, Database, FileStack, Loader2, MemoryStick, Network, TriangleAlert, Wrench } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { RuntimeEvent, RuntimeEventType } from "@/types/agent";

const icons: Record<RuntimeEventType, typeof Brain> = {
  thinking: Brain,
  retrieval: Database,
  chunk: FileStack,
  tool_call: Wrench,
  mcp: Network,
  memory: MemoryStick,
  compression: FileStack,
  metrics: CheckCircle2,
  error: TriangleAlert,
};

export function PipelineTimeline({ events }: { events: RuntimeEvent[] }) {
  if (events.length === 0) {
    return (
      <div className="runtime-grid rounded-lg border p-6 text-center">
        <Network className="mx-auto mb-3 h-5 w-5 text-zinc-500" />
        <div className="text-sm font-medium text-zinc-300">Runtime graph is idle</div>
        <div className="mt-1 text-xs leading-5 text-zinc-500">Pipeline phases appear here as the agent thinks, retrieves, calls tools, and updates memory.</div>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {events.map((event) => {
        const Icon = icons[event.type];
        return (
          <div key={event.id} className="rounded-lg border bg-zinc-950/50 p-3">
            <div className="flex items-start gap-3">
              <div className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-md border border-zinc-700 bg-zinc-900">
                {event.status === "running" ? <Loader2 className="h-3.5 w-3.5 animate-spin text-cyan-200" /> : <Icon className="h-3.5 w-3.5 text-zinc-300" />}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-2">
                  <div className="truncate text-sm font-medium text-zinc-100">{event.title}</div>
                  <Badge tone={event.status === "error" ? "rose" : event.status === "running" ? "cyan" : "zinc"}>{event.status}</Badge>
                </div>
                {event.detail && <div className="mt-1 line-clamp-3 text-xs leading-5 text-zinc-500">{event.detail}</div>}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
