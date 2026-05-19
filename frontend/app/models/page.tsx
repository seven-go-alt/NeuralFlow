import { Cpu, Sparkles, Zap } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { listModels } from "@/services/models";

export default async function ModelsPage() {
  const data = await listModels().catch(() => ({ models: [], current_model: "" }));

  const { current_model, models } = data;

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
      <div className="mx-auto max-w-5xl space-y-8">
        {/* Header */}
        <div>
          <div className="text-2xl font-semibold">Models</div>
          <div className="mt-1 text-sm text-zinc-500">
            Available language models for the agent runtime
          </div>
        </div>

        {/* Current Model */}
        <section>
          <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-zinc-500 font-mono">
            <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
            Current Model
          </div>
          {current_model ? (
            <div className="stagger-children">
              <div className="rounded-xl border border-cyan-400/25 bg-gradient-to-br from-cyan-400/10 to-zinc-900/60 p-5 animate-fade-in-up">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="grid h-10 w-10 place-items-center rounded-lg border border-cyan-400/30 bg-cyan-400/15">
                      <Cpu className="h-5 w-5 text-cyan-200" />
                    </div>
                    <div>
                      <div className="text-base font-semibold text-zinc-50">{current_model}</div>
                      <div className="mt-0.5 text-xs text-zinc-500 font-mono">Active inference engine</div>
                    </div>
                  </div>
                  <Badge tone="cyan" pulse>
                    live
                  </Badge>
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-lg border border-dashed border-zinc-700 p-12 text-center text-sm text-zinc-500">
              No model currently selected
            </div>
          )}
        </section>

        {/* Available Models */}
        <section>
          <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-zinc-500 font-mono">
            <Zap className="h-3.5 w-3.5 text-zinc-400" />
            Available Models
          </div>
          {models.length === 0 ? (
            <div className="rounded-lg border border-dashed border-zinc-700 p-12 text-center text-sm text-zinc-500">
              No models available. Ensure the backend is running and configured.
            </div>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {models.map((model, idx) => (
                <div
                  key={model}
                  className={`animate-fade-in-up rounded-lg border p-4 transition-colors ${
                    model === current_model
                      ? "border-cyan-400/30 bg-cyan-400/8"
                      : "border-zinc-800 bg-zinc-900/50 hover:bg-zinc-800/50"
                  }`}
                  style={{ animationDelay: `${idx * 0.05}s` }}
                >
                  <div className="flex items-start justify-between">
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium text-zinc-100 font-mono">
                        <span className="font-sans">{model}</span>
                      </div>
                    </div>
                    {model === current_model && (
                      <Badge tone="cyan" className="ml-2 shrink-0">
                        active
                      </Badge>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
