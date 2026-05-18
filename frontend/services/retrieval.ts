import type { RetrievalResponse } from "@/types/rag";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? (typeof window !== "undefined" ? `${window.location.protocol}//${window.location.hostname}:20004` : "http://localhost:8000");

export async function searchRetrieval(query: string, options?: { documentIds?: string[] }): Promise<RetrievalResponse> {
  const response = await fetch(`${API_BASE}/api/v1/retrieval/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      top_k: 5,
      score_threshold: 0.2,
      filters: { document_ids: options?.documentIds ?? [] },
    }),
    cache: "no-store",
  });
  if (!response.ok) throw new Error("Failed to search retrieval");
  return response.json();
}
