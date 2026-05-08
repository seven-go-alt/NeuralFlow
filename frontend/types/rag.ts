export interface RetrievalSource {
  title?: string | null;
  filename?: string | null;
  page_number?: number | null;
}

export interface RetrievalResult {
  chunk_id: string;
  document_id: string;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
  source: RetrievalSource;
}

export interface RetrievalResponse {
  query: string;
  results: RetrievalResult[];
}
