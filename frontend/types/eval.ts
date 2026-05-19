export interface EvalRunSummary {
  run_id: string;
  dataset_name: string;
  total_cases: number;
  retrieval_hit_rate: number;
  citation_accuracy: number;
  keyword_coverage: number;
  average_latency_ms: number;
  started_at: string;
  completed_at: string | null;
  answer_relevance: number | null;
  answer_faithfulness: number | null;
  answer_completeness: number | null;
}

export interface EvalRunsResponse {
  runs: EvalRunSummary[];
}

export interface EvalRunDetail {
  run_id: string;
  dataset_name: string;
  total_cases: number;
  metrics: Record<string, unknown>;
  per_case_results: Record<string, unknown>;
  started_at: string;
  completed_at: string | null;
}
