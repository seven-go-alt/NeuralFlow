import type { ModelsResponse } from "@/types/agent";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  (typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:20004`
    : "http://localhost:8000");

export async function listModels(): Promise<ModelsResponse> {
  const response = await fetch(`${API_BASE}/api/v1/models`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to list models: ${response.status}`);
  return response.json();
}
