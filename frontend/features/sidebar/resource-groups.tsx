"use client";

import { Database, Plug, Settings, ShieldCheck } from "lucide-react";
import { useQuery } from "@tanstack/react-query";

import { Badge } from "@/components/ui/badge";
import { apiClient } from "@/services/api-client";
import { useAgentStore } from "@/store/agent-store";

const settingsLinks = [
  { label: "Tenant isolation", icon: ShieldCheck },
  { label: "Runtime config", icon: Settings },
];

export function ResourceGroups() {
  const apiBaseUrl = useAgentStore((state) => state.apiBaseUrl);
  const { data: skills = [] } = useQuery({ queryKey: ["skills", apiBaseUrl], queryFn: () => apiClient(apiBaseUrl).skills() });

  return (
    <div className="space-y-5">
      <section>
        <div className="mb-2 flex items-center justify-between">
          <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Knowledge Bases</div>
          <Badge tone="emerald">RAG</Badge>
        </div>
        {["Working memory", "Long-term memory", "Vector archive"].map((item, index) => (
          <div key={item} className="mb-1 flex items-center justify-between rounded-md border border-transparent px-2 py-1.5 text-sm text-zinc-300 hover:border-zinc-800 hover:bg-zinc-900">
            <span className="flex items-center gap-2">
              <Database className={`h-3.5 w-3.5 ${index === 0 ? "text-emerald-300" : index === 1 ? "text-violet-300" : "text-cyan-300"}`} />
              {item}
            </span>
            <span className="text-[11px] text-zinc-500">live</span>
          </div>
        ))}
      </section>
      <section>
        <div className="mb-2 flex items-center justify-between">
          <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Tools</div>
          <Badge tone="cyan">{skills.length}</Badge>
        </div>
        <div className="space-y-1">
          {skills.slice(0, 6).map((skill) => (
            <div key={skill.name} className="rounded-md border border-zinc-800 bg-zinc-950/40 p-2">
              <div className="flex items-center gap-2 text-sm text-zinc-200">
                <Plug className="h-3.5 w-3.5 text-cyan-300" />
                {skill.name}
              </div>
              <p className="mt-1 line-clamp-2 text-[11px] leading-4 text-zinc-500">{skill.description}</p>
            </div>
          ))}
          {skills.length === 0 && <div className="text-xs leading-5 text-zinc-500">No tools reported yet. Start the FastAPI backend to populate this list.</div>}
        </div>
      </section>
      <section>
        <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">Settings</div>
        {settingsLinks.map(({ label, icon: Icon }) => (
          <div key={label} className="mb-1 flex items-center gap-2 rounded-md px-2 py-1.5 text-sm text-zinc-300 hover:bg-zinc-900">
            <Icon className="h-3.5 w-3.5 text-violet-300" />
            {label}
          </div>
        ))}
      </section>
    </div>
  );
}
