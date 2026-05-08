export interface DocumentItem {
  document_id: string;
  tenant_id: string;
  owner_user_id: string;
  title?: string | null;
  filename: string;
  original_filename: string;
  file_type: string;
  mime_type: string;
  size_bytes: number;
  storage_path: string;
  checksum_sha256: string;
  status: string;
  chunk_count: number;
  token_count?: number | null;
  metadata_json: Record<string, unknown>;
  source_info_json: Record<string, unknown>;
  error_message?: string | null;
  failed_stage?: string | null;
  created_at: string;
  updated_at: string;
  indexed_at?: string | null;
}

export interface DocumentListResponse {
  items: DocumentItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface DocumentChunkItem {
  chunk_id: string;
  document_id: string;
  tenant_id: string;
  chunk_index: number;
  content: string;
  token_count: number;
  page_number?: number | null;
  section_title?: string | null;
  metadata_json: Record<string, unknown>;
  embedding_model?: string | null;
  embedding_status: string;
  created_at: string;
}

export interface DocumentChunksResponse {
  items: DocumentChunkItem[];
  total: number;
}
