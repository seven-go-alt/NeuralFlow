import type { RetrievalResponse } from "@/types/rag";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function searchRetrieval(query: string): Promise<RetrievalResponse> {
  const response = await fetch(`${API_BASE}/api/retrieval/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 5, score_threshold: 0.2, filters: {} }),
    cache: "no-store",
  });
  if (!response.ok) throw new Error("Failed to search retrieval");
  return response.json();
}
