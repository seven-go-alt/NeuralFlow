export interface TraceSummary {
  trace_id: string;
  session_id: string;
  query: string;
  total_duration_ms: number;
  token_count: number | null;
  created_at: string;
}

export interface TracesResponse {
  traces: TraceSummary[];
}

export interface TraceDetail {
  trace_id: string;
  tenant_id: string;
  session_id: string;
  query: string;
  answer: string | null;
  span_tree: Record<string, unknown>;
  total_duration_ms: number;
  token_count: number | null;
  created_at: string;
}
