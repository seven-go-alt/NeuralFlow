import type { DocumentChunksResponse, DocumentItem, DocumentListResponse } from "@/types/documents";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? (typeof window !== "undefined" ? `${window.location.protocol}//${window.location.hostname}:20004` : "http://localhost:8000");

export async function listDocuments(): Promise<DocumentListResponse> {
  const response = await fetch(`${API_BASE}/api/documents`, { cache: "no-store" });
  if (!response.ok) throw new Error("Failed to load documents");
  return response.json();
}

export async function getDocument(documentId: string): Promise<DocumentItem> {
  const response = await fetch(`${API_BASE}/api/documents/${documentId}`, { cache: "no-store" });
  if (!response.ok) throw new Error("Failed to load document");
  return response.json();
}

export async function getDocumentChunks(documentId: string): Promise<DocumentChunksResponse> {
  const response = await fetch(`${API_BASE}/api/documents/${documentId}/chunks`, { cache: "no-store" });
  if (!response.ok) throw new Error("Failed to load document chunks");
  return response.json();
}

export async function reindexDocument(documentId: string): Promise<{ ok: boolean; document_id: string; status: string }> {
  const response = await fetch(`${API_BASE}/api/documents/${documentId}/reindex`, {
    method: "POST",
    cache: "no-store",
  });
  if (!response.ok) throw new Error("Failed to reindex document");
  return response.json();
}
