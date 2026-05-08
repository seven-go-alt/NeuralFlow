export type ChatMode = "stream" | "react" | "orchestrate";

export type MessageRole = "user" | "assistant" | "system";

export type RuntimeStatus = "pending" | "running" | "success" | "error" | "idle";

export type RuntimeEventType =
  | "thinking"
  | "retrieval"
  | "chunk"
  | "tool_call"
  | "mcp"
  | "memory"
  | "compression"
  | "metrics"
  | "error";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: number;
  status?: RuntimeStatus;
  tokens?: number;
  latencyMs?: number;
  intent?: string;
  usedSkills?: string[];
}

export interface ConversationSession {
  id: string;
  title: string;
  updatedAt: number;
  model: string;
  messageCount: number;
}

export interface RuntimeEvent {
  id: string;
  type: RuntimeEventType;
  title: string;
  detail?: string;
  status: RuntimeStatus;
  timestamp: number;
  latencyMs?: number;
  payload?: unknown;
}

export interface RetrievedChunk {
  id: string;
  source: string;
  score: number;
  text: string;
}

export interface ToolCall {
  id: string;
  name: string;
  status: RuntimeStatus;
  input?: unknown;
  output?: unknown;
  latencyMs?: number;
}

export interface RuntimeMetrics {
  tokensIn?: number;
  tokensOut?: number;
  latencyMs?: number;
  retrievalMs?: number;
  toolMs?: number;
}

export interface RuntimeSnapshot {
  events: RuntimeEvent[];
  retrievedChunks: RetrievedChunk[];
  toolCalls: ToolCall[];
  metrics: RuntimeMetrics;
}

export interface Skill {
  name: string;
  description: string;
}

export interface ModelsResponse {
  models: string[];
  current_model: string;
  error?: string;
}

export interface ChatResponse {
  session_id: string;
  intent: string;
  prompt: string;
  reply: string;
  used_skills: string[];
  skill_results: Array<{ skill: string; result: unknown }>;
}

export interface ReactAgentResponse {
  session_id: string;
  intent?: string;
  route?: string;
  route_reason?: string;
  final_answer: string;
  steps: Array<Record<string, unknown>>;
  total_iterations: number;
  reflections?: string[];
}
