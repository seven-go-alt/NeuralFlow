import type { TraceDetail, TracesResponse } from "@/types/traces";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  (typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:20004`
    : "http://localhost:8000");

export async function listTraces(): Promise<TracesResponse> {
  const response = await fetch(`${API_BASE}/api/v1/traces`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to list traces: ${response.status}`);
  return response.json();
}

export async function getTrace(traceId: string): Promise<TraceDetail> {
  const response = await fetch(`${API_BASE}/api/v1/traces/${traceId}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to get trace: ${response.status}`);
  return response.json();
}
