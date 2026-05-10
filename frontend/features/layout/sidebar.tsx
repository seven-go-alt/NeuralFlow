"use client";

import Link from "next/link";
import { Bot, Cpu, GitBranch, Radio, Route } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { useAgentStore } from "@/store/agent-store";
import type { ChatMode } from "@/types/agent";
import { ModelSelector } from "@/features/sidebar/model-selector";
import { ResourceGroups } from "@/features/sidebar/resource-groups";
import { SessionList } from "@/features/sidebar/session-list";

const modes: Array<{ value: ChatMode; label: string; description: string; icon: typeof Bot }> = [
  { value: "stream", label: "Streaming", description: "SSE tokens with reasoning deltas", icon: Radio },
  { value: "react", label: "ReAct", description: "Function calling tool loop", icon: GitBranch },
  { value: "orchestrate", label: "Orchestrate", description: "Route to specialist agents", icon: Route },
];

export function Sidebar() {
  const mode = useAgentStore((state) => state.mode);
  const setMode = useAgentStore((state) => state.setMode);

  return (
    <aside className="hairline-panel hidden w-[292px] shrink-0 flex-col border-r bg-zinc-950/80 backdrop-blur md:flex">
      <div className="flex h-16 items-center gap-2 border-b px-4">
        <div className="grid h-9 w-9 place-items-center rounded-lg border border-cyan-300/30 bg-cyan-300/15 text-sm font-bold text-cyan-100">N</div>
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold text-zinc-50">NeuralFlow</div>
          <div className="text-[11px] text-zinc-500">Agent Runtime Platform</div>
        </div>
        <Badge tone="emerald" className="ml-auto">
          live
        </Badge>
      </div>
      <div className="flex-1 space-y-6 overflow-y-auto p-4">
        <div className="rounded-xl border border-cyan-300/20 bg-cyan-300/10 p-3">
          <div className="flex items-center gap-2 text-sm font-medium text-cyan-100">
            <Cpu className="h-4 w-4" />
            Runtime cluster
          </div>
          <div className="mt-3 grid grid-cols-3 gap-2 text-center">
            {[
              ["12", "runs"],
              ["4", "tools"],
              ["8k", "ctx"],
            ].map(([value, label]) => (
              <div key={label} className="rounded-md border border-white/10 bg-black/20 px-2 py-2">
                <div className="text-sm font-semibold text-zinc-100">{value}</div>
                <div className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
              </div>
            ))}
          </div>
        </div>
        <SessionList />
        <ModelSelector />
        <section>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">Execution Mode</div>
          <div className="space-y-1">
            {modes.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.value}
                  onClick={() => setMode(item.value)}
                  className={`w-full rounded-md border p-3 text-left transition-colors ${
                    mode === item.value ? "border-cyan-400/40 bg-cyan-400/10" : "border-zinc-800 bg-zinc-950/40 hover:bg-zinc-900"
                  }`}
                >
                  <div className="flex items-center gap-2 text-sm font-medium text-zinc-100">
                    <Icon className="h-4 w-4 text-cyan-200" />
                    {item.label}
                  </div>
                  <div className="mt-1 text-[11px] leading-4 text-zinc-500">{item.description}</div>
                </button>
              );
            })}
          </div>
        </section>
        <ResourceGroups />
        <section>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">Knowledge Base</div>
          <Link href="/documents" className="block rounded-md border border-zinc-800 bg-zinc-950/40 px-3 py-3 text-sm text-zinc-200 transition-colors hover:bg-zinc-900">
            Documents / RAG Console
          </Link>
        </section>
      </div>
    </aside>
  );
}
