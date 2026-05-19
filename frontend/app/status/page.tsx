"use client";

import { useEffect, useRef, useState } from "react";
import { Activity, Clock, Database, RefreshCw, Server } from "lucide-react";

import { Badge } from "@/components/ui/badge";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  (typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:20004`
    : "http://localhost:8000");

interface ComponentStatus {
  status: string;
  [key: string]: unknown;
}

interface HealthData {
  status: string;
  components: Record<string, ComponentStatus>;
  response_time_ms?: number;
  [key: string]: unknown;
}

type BadgeTone = "cyan" | "emerald" | "amber" | "rose" | "violet" | "zinc";

function statusTone(status: string): BadgeTone {
  const s = status.toLowerCase();
  if (s === "healthy" || s === "ok" || s === "connected") return "emerald";
  if (s === "degraded" || s === "slow") return "amber";
  if (s === "unhealthy" || s === "error" || s === "disconnected") return "rose";
  return "zinc";
}

function statusLabel(status: string): string {
  const s = status.toLowerCase();
  if (s === "healthy" || s === "ok") return "Healthy";
  if (s === "degraded") return "Degraded";
  if (s === "unhealthy" || s === "error") return "Unhealthy";
  return status;
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatRelativeTime(date: Date): string {
  const diff = Date.now() - date.getTime();
  if (diff < 1000) return "just now";
  if (diff < 60_000) return `${Math.floor(diff / 1000)}s ago`;
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  return `${Math.floor(diff / 3_600_000)}h ago`;
}

const componentIcons: Record<string, typeof Database> = {
  database: Database,
  chromadb: Database,
  redis: Server,
};

function componentIcon(name: string) {
  return componentIcons[name] ?? Server;
}

export default function StatusPage() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshKey, setRefreshKey] = useState(0);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const historyRef = useRef<string[]>([]);

  useEffect(() => {
    let cancelled = false;

    const fetchHealth = async () => {
      const start = performance.now();
      try {
        const res = await fetch(`${API_BASE}/healthz`, { cache: "no-store" });
        const data: HealthData = await res.json();
        data.response_time_ms = Math.round(performance.now() - start);
        if (!cancelled) {
          setHealth(data);
          setError(null);
          const now = new Date();
          setLastChecked(now);

          // Track status history (keep last 10)
          historyRef.current = [
            `${formatTime(now)} - ${data.status}`,
            ...historyRef.current.slice(0, 9),
          ];
          setHistory([...historyRef.current]);
        }
      } catch {
        if (!cancelled) setError("Failed to reach the API health endpoint");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 15_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [refreshKey]);

  const handleRefresh = () => {
    setLoading(true);
    setRefreshKey((k) => k + 1);
  };

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
      <div className="mx-auto max-w-4xl space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <Activity className="h-6 w-6 text-cyan-400" />
              <div className="text-2xl font-semibold">API Status</div>
            </div>
            <div className="mt-1 text-sm text-zinc-500">
              System health dashboard with automatic refresh every 15s
            </div>
          </div>
          <button
            onClick={handleRefresh}
            disabled={loading}
            className="flex items-center gap-2 rounded-md border border-zinc-800 bg-zinc-950/40 px-3 py-2 text-xs text-zinc-400 transition-colors hover:bg-zinc-900 hover:text-zinc-200 font-mono disabled:opacity-40"
          >
            <RefreshCw
              className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`}
            />
            Refresh
          </button>
        </div>

        {/* Overall Status */}
        {health && (
          <section className="stagger-children space-y-4">
            <div className="animate-fade-in-up rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div
                    className={`grid h-10 w-10 place-items-center rounded-lg border ${
                      health.status === "healthy" || health.status === "ok"
                        ? "border-emerald-400/30 bg-emerald-400/15"
                        : "border-rose-400/30 bg-rose-400/15"
                    }`}
                  >
                    <Activity
                      className={`h-5 w-5 ${
                        health.status === "healthy" || health.status === "ok"
                          ? "text-emerald-200"
                          : "text-rose-200"
                      }`}
                    />
                  </div>
                  <div>
                    <div className="text-sm font-medium text-zinc-100">
                      Overall System Health
                    </div>
                    <div className="mt-0.5 text-xs text-zinc-500 font-mono">
                      Response time: {health.response_time_ms ?? "—"}ms
                    </div>
                  </div>
                </div>
                <Badge
                  tone={statusTone(health.status)}
                  pulse={
                    health.status === "healthy" || health.status === "ok"
                  }
                >
                  {statusLabel(health.status)}
                </Badge>
              </div>
              {lastChecked && (
                <div className="mt-3 flex items-center gap-1.5 text-xs text-zinc-600">
                  <Clock className="h-3 w-3" />
                  Last checked: {formatTime(lastChecked)} (
                  {formatRelativeTime(lastChecked)})
                </div>
              )}
            </div>
          </section>
        )}

        {/* Error State */}
        {error && (
          <div className="animate-fade-in-up rounded-lg border border-rose-800/50 bg-rose-950/20 p-4">
            <p className="text-sm text-rose-300 font-mono">{error}</p>
          </div>
        )}

        {/* Loading State */}
        {loading && !health && !error && (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="animate-pulse rounded-lg border border-zinc-800 bg-zinc-900/30 p-5"
              >
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-lg bg-zinc-800" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 w-32 rounded bg-zinc-800" />
                    <div className="h-3 w-24 rounded bg-zinc-800/60" />
                  </div>
                  <div className="h-6 w-16 rounded-md bg-zinc-800" />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Component Statuses */}
        {health?.components && (
          <section className="stagger-children space-y-3">
            <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-zinc-500 font-mono">
              <Server className="h-3.5 w-3.5 text-zinc-400" />
              Components
            </div>
            {Object.entries(health.components).map(([name, comp], idx) => {
              const Icon = componentIcon(name);
              const compStatus =
                typeof comp === "string" ? comp : (comp.status ?? "unknown");
              return (
                <div
                  key={name}
                  className="animate-fade-in-up rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 transition-colors hover:bg-zinc-800/50"
                  style={{ animationDelay: `${idx * 0.06}s` }}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="grid h-9 w-9 place-items-center rounded-lg border border-zinc-700 bg-zinc-900">
                        <Icon className="h-4 w-4 text-zinc-400" />
                      </div>
                      <div>
                        <div className="text-sm font-medium capitalize text-zinc-100">
                          {name}
                        </div>
                        <div className="mt-0.5 text-xs text-zinc-600 font-mono">
                          {compStatus}
                        </div>
                      </div>
                    </div>
                    <Badge tone={statusTone(compStatus)}>
                      {statusLabel(compStatus)}
                    </Badge>
                  </div>
                </div>
              );
            })}
          </section>
        )}

        {/* Status History */}
        {history.length > 0 && (
          <section>
            <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-zinc-500 font-mono">
              <Clock className="h-3.5 w-3.5 text-zinc-400" />
              Status History
            </div>
            <div className="animate-fade-in-up rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
              <div className="space-y-1.5">
                {history.map((entry, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-2 text-xs font-mono text-zinc-500"
                  >
                    <span className="shrink-0 text-zinc-600">{entry}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {/* No data */}
        {!health && !loading && !error && (
          <div className="rounded-lg border border-dashed border-zinc-700 p-12 text-center text-sm text-zinc-500">
            No health data available. Ensure the backend is running.
          </div>
        )}
      </div>
    </main>
  );
}
