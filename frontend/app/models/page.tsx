"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, Cpu, RefreshCw, Sparkles, Zap } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { ModelsResponse } from "@/types/agent";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  (typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:20004`
    : "http://localhost:8000");

export default function ModelsPage() {
  const [data, setData] = useState<ModelsResponse>({
    models: [],
    current_model: "",
  });
  const [loading, setLoading] = useState(true);
  const [switching, setSwitching] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const doFetchModels = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/models`, {
        cache: "no-store",
      });
      const json: ModelsResponse = await res.json();
      setData(json);
      setError(null);
    } catch {
      setError("Failed to load models");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const fetchData = async () => {
      if (cancelled) return;
      await doFetchModels();
    };
    fetchData();
    return () => { cancelled = true; };
  }, []);

  async function switchModel(model: string) {
    if (model === data.current_model) return;
    setSwitching(model);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/models/switch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model }),
      });
      if (!res.ok) throw new Error(`Switch failed: ${res.status}`);
      await doFetchModels();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to switch model");
    } finally {
      setSwitching(null);
    }
  }

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

        {/* Loading State */}
        {loading && (
          <div className="space-y-3">
            <div className="animate-pulse rounded-xl border border-zinc-800 bg-zinc-900/30 p-5">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-lg bg-zinc-800" />
                <div className="flex-1 space-y-2">
                  <div className="h-4 w-48 rounded bg-zinc-800" />
                  <div className="h-3 w-32 rounded bg-zinc-800/60" />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && !loading && (
          <div className="rounded-lg border border-rose-800/50 bg-rose-950/20 p-4">
            <p className="text-sm text-rose-300 font-mono">{error}</p>
          </div>
        )}

        {/* Current Model */}
        <section>
          <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-zinc-500 font-mono">
            <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
            Current Model
          </div>
          {current_model ? (
            <div className="stagger-children">
              <div className="animate-fade-in-up rounded-xl border border-cyan-400/25 bg-gradient-to-br from-cyan-400/10 to-zinc-900/60 p-5">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="grid h-10 w-10 place-items-center rounded-lg border border-cyan-400/30 bg-cyan-400/15">
                      <Cpu className="h-5 w-5 text-cyan-200" />
                    </div>
                    <div>
                      <div className="text-base font-semibold text-zinc-50">
                        {current_model}
                      </div>
                      <div className="mt-0.5 text-xs text-zinc-500 font-mono">
                        Active inference engine
                      </div>
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
              {models.map((model, idx) => {
                const isActive = model === current_model;
                const isSwitching = switching === model;
                return (
                  <button
                    key={model}
                    onClick={() => switchModel(model)}
                    disabled={isActive || switching !== null}
                    className={`animate-fade-in-up rounded-lg border p-4 text-left transition-all ${
                      isActive
                        ? "cursor-default border-cyan-400/30 bg-cyan-400/8"
                        : "cursor-pointer border-zinc-800 bg-zinc-900/50 hover:border-zinc-700 hover:bg-zinc-800/50"
                    } ${isSwitching ? "opacity-70" : ""}`}
                    style={{ animationDelay: `${idx * 0.05}s` }}
                  >
                    <div className="flex items-start justify-between">
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-medium text-zinc-100 font-mono">
                          {model}
                        </div>
                        {isSwitching && (
                          <div className="mt-1 flex items-center gap-1.5 text-xs text-cyan-300">
                            <RefreshCw className="h-3 w-3 animate-spin" />
                            Switching...
                          </div>
                        )}
                        {isActive && !isSwitching && (
                          <div className="mt-1 flex items-center gap-1.5 text-xs text-emerald-400">
                            <CheckCircle2 className="h-3 w-3" />
                            Active
                          </div>
                        )}
                      </div>
                      {isActive && !isSwitching && (
                        <Badge tone="cyan" className="ml-2 shrink-0">
                          active
                        </Badge>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
          {error && (
            <div className="mt-4 rounded-md border border-rose-800/30 bg-rose-950/15 p-3 text-xs text-rose-300">
              {error}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
