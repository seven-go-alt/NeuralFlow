"use client";

import { MessageSquarePlus } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useAgentStore } from "@/store/agent-store";

export function SessionList() {
  const sessions = useAgentStore((state) => state.sessions);
  const activeSessionId = useAgentStore((state) => state.activeSessionId);
  const setActiveSession = useAgentStore((state) => state.setActiveSession);
  const createSession = useAgentStore((state) => state.createSession);

  return (
    <section className="min-h-0">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Sessions</div>
        <Button size="icon" variant="ghost" title="New session" onClick={createSession}>
          <MessageSquarePlus className="h-4 w-4" />
        </Button>
      </div>
      <div className="space-y-1">
        {sessions.map((session) => (
          <button
            key={session.id}
            onClick={() => setActiveSession(session.id)}
            className={`w-full rounded-md border px-3 py-2 text-left transition-colors ${
              session.id === activeSessionId ? "border-cyan-400/40 bg-cyan-400/10" : "border-transparent hover:border-zinc-800 hover:bg-zinc-900"
            }`}
          >
            <div className="truncate text-sm text-zinc-100">{session.title}</div>
            <div className="mt-1 flex justify-between text-[11px] text-zinc-500">
              <span>{session.messageCount} messages</span>
              <span>{session.model}</span>
            </div>
          </button>
        ))}
      </div>
    </section>
  );
}
