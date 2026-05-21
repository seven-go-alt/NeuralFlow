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

export interface CaseResult {
  case_id: string;
  question: string;
  retrieved_doc_ids: string[];
  retrieved_contents: string[];
  answer: string | null;
  latency_ms: number;
  retrieval_hit: boolean;
  citation_match: boolean;
  keyword_coverage: number;
  no_answer_correct: boolean | null;
  answer_relevance: number | null;
  answer_faithfulness: number | null;
  answer_completeness: number | null;
  first_relevant_rank: number;
  precision_at_k: number;
  recall_at_k: number;
}

export interface PerCaseResults {
  results: CaseResult[];
}

export interface EvalMetrics {
  retrieval_hit_rate: number;
  citation_accuracy: number;
  keyword_coverage: number;
  average_latency_ms: number;
  answer_relevance: number | null;
  answer_faithfulness: number | null;
  answer_completeness: number | null;
}

export interface EvalRunDetail {
  run_id: string;
  dataset_name: string;
  total_cases: number;
  metrics: EvalMetrics;
  per_case_results: PerCaseResults;
  started_at: string;
  completed_at: string | null;
}

export interface TriggerEvalRunRequest {
  dataset_path: string;
  top_k?: number;
}

export interface TriggerEvalRunResponse {
  run_id: string;
  status: string;
  total_cases: number;
}
