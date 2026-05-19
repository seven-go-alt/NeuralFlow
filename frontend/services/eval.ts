import type { EvalRunDetail, EvalRunsResponse } from "@/types/eval";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  (typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:20004`
    : "http://localhost:8000");

export async function listEvalRuns(): Promise<EvalRunsResponse> {
  const response = await fetch(`${API_BASE}/api/v1/eval/runs`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to list eval runs: ${response.status}`);
  return response.json();
}

export async function getEvalRun(runId: string): Promise<EvalRunDetail> {
  const response = await fetch(`${API_BASE}/api/v1/eval/runs/${runId}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to get eval run: ${response.status}`);
  return response.json();
}
