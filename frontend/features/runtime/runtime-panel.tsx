"use client";

import { Activity, Brain, Database, FileStack, MemoryStick, PanelRightClose, Wrench } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAgentStore } from "@/store/agent-store";

import { MetricsStrip } from "./metrics-strip";
import { PipelineTimeline } from "./pipeline-timeline";
import { RetrievedChunks } from "./retrieved-chunks";
import { ToolCallList } from "./tool-call-list";

export function RuntimePanel() {
  const runtime = useAgentStore((state) => state.runtime);
  const toggleRightPanel = useAgentStore((state) => state.toggleRightPanel);

  return (
    <aside className="hairline-panel hidden w-[360px] shrink-0 border-l bg-zinc-950/80 backdrop-blur lg:flex lg:flex-col xl:w-[400px] animate-slide-in-right">
      <div className="flex h-16 items-center justify-between border-b px-4">
        <div className="flex items-center gap-2">
          <div className="grid h-9 w-9 place-items-center rounded-lg border border-cyan-300/30 bg-cyan-300/10">
            <Activity className="h-4 w-4 text-cyan-200" />
          </div>
          <div>
            <div className="text-sm font-semibold font-mono">Runtime</div>
            <div className="text-[11px] text-zinc-500 font-mono">Reasoning · RAG · MCP · Memory</div>
          </div>
        </div>
        <Button variant="ghost" size="icon" onClick={toggleRightPanel} title="Hide runtime panel">
          <PanelRightClose className="h-4 w-4" />
        </Button>
      </div>
      <div className="flex-1 space-y-5 overflow-y-auto p-4 stagger-children">
        <div className="rounded-xl border bg-zinc-950/55 p-3">
          <div className="mb-3 flex items-center justify-between">
            <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500 font-mono">Live topology</div>
            <Badge tone="emerald" pulse className="font-mono">ready</Badge>
          </div>
          <div className="grid grid-cols-5 items-center gap-2">
            {[
              { label: "Think", icon: Brain, color: "text-violet-300" },
              { label: "RAG", icon: Database, color: "text-emerald-300" },
              { label: "MCP", icon: Wrench, color: "text-amber-300" },
              { label: "Mem", icon: MemoryStick, color: "text-fuchsia-300" },
              { label: "Out", icon: FileStack, color: "text-cyan-300" },
            ].map(({ label, icon: Icon, color }) => (
              <div key={label} className="rounded-lg border bg-black/20 p-2 text-center">
                <Icon className={`mx-auto h-4 w-4 ${color}`} />
                <div className="mt-1 text-[10px] text-zinc-500">{label}</div>
              </div>
            ))}
          </div>
        </div>
        <MetricsStrip metrics={runtime.metrics} />
        <PipelineTimeline events={runtime.events} />
        <RetrievedChunks chunks={runtime.retrievedChunks} />
        <ToolCallList toolCalls={runtime.toolCalls} />
      </div>
    </aside>
  );
}
