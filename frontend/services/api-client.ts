import type { ChatResponse, ModelsResponse, ReactAgentResponse, Skill } from "@/types/agent";

export class ApiClient {
  constructor(private readonly baseUrl: string) {}

  async health() {
    return this.get<{ status: string; app: string }>("/healthz");
  }

  async skills() {
    const data = await this.get<{ skills: Skill[] }>("/api/skills");
    return data.skills;
  }

  async models() {
    return this.get<ModelsResponse>("/api/models");
  }

  async switchModel(model: string) {
    return this.post<{ model: string; message: string }>("/api/models/switch", { model });
  }

  async chat(sessionId: string, message: string, options?: { documentIds?: string[] }) {
    return this.post<ChatResponse>("/chat", {
      session_id: sessionId,
      message,
      use_retrieval: true,
      retrieval_options: { filters: { document_ids: options?.documentIds ?? [] } },
    });
  }

  async react(sessionId: string, message: string) {
    return this.post<ReactAgentResponse>("/chat/react", { session_id: sessionId, message });
  }

  async orchestrate(sessionId: string, message: string) {
    return this.post<ReactAgentResponse>("/chat/orchestrate", { session_id: sessionId, message });
  }

  private async get<T>(path: string): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`${path} failed: ${response.status}`);
    return response.json() as Promise<T>;
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(`${path} failed: ${response.status}`);
    return response.json() as Promise<T>;
  }
}

export function apiClient(baseUrl: string) {
  return new ApiClient(baseUrl.replace(/\/$/, ""));
}
