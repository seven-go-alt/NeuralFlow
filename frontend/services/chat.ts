import type { ChatResponse } from "@/types/agent";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  (typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:20004`
    : "http://localhost:8000");

export interface ChatStreamCallbacks {
  onToken: (token: string) => void;
  onDone: (sessionId: string) => void;
  onError: (error: Error) => void;
}

export async function sendChatMessage(
  message: string,
  sessionId: string | null,
  callbacks: ChatStreamCallbacks,
  signal?: AbortSignal,
): Promise<string> {
  const response = await fetch(`${API_BASE}/api/v1/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      stream: true,
    }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body stream");

  const decoder = new TextDecoder();
  let buffer = "";
  let newSessionId = sessionId ?? "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data: ")) continue;

      const payload = trimmed.slice(6).trim();

      if (payload === "[DONE]") {
        callbacks.onDone(newSessionId);
        continue;
      }

      try {
        const parsed = JSON.parse(payload);
        if (parsed.session_id) {
          newSessionId = parsed.session_id;
        }
        if (parsed.type === "error" || parsed.error) {
          callbacks.onError(new Error(parsed.error ?? parsed.message ?? "Stream error"));
        } else if (parsed.type === "token" || parsed.token !== undefined) {
          callbacks.onToken(parsed.token ?? parsed.content ?? "");
        } else if (parsed.type === "done") {
          callbacks.onDone(newSessionId);
        }
      } catch {
        callbacks.onToken(payload);
      }
    }
  }

  return newSessionId;
}

export async function sendChatMessageNonStreaming(
  message: string,
  sessionId: string | null,
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/api/v1/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      stream: false,
    }),
  });

  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  return response.json();
}
