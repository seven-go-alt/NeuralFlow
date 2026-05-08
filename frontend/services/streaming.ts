export interface StreamCallbacks {
  onEvent?: (event: string, data: Record<string, unknown>) => void;
  onDelta?: (delta: string) => void;
  onThinking?: (delta: string) => void;
  onDone?: (data: Record<string, unknown>) => void;
  onError?: (error: Error) => void;
  onRetrieval?: (data: Record<string, unknown>) => void;
  onChunk?: (data: Record<string, unknown>) => void;
}

export async function streamChat({
  baseUrl,
  sessionId,
  message,
  signal,
  callbacks,
}: {
  baseUrl: string;
  sessionId: string;
  message: string;
  signal?: AbortSignal;
  callbacks: StreamCallbacks;
}) {
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/chat/stream?include_thinking=true`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ session_id: sessionId, message }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Stream failed: ${response.status}`);
  }

  if (!response.body) {
    throw new Error("ReadableStream is not supported by this browser");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const parsed = parseSseFrame(frame);
      if (!parsed) continue;
      callbacks.onEvent?.(parsed.event, parsed.data);

      const delta = typeof parsed.data.delta === "string" ? parsed.data.delta : "";
      if (parsed.event === "message") callbacks.onDelta?.(delta);
      if (parsed.event === "thinking") callbacks.onThinking?.(delta);
      if (parsed.event === "done") callbacks.onDone?.(parsed.data);
      if (parsed.event === "retrieval") callbacks.onRetrieval?.(parsed.data);
      if (parsed.event === "chunk") callbacks.onChunk?.(parsed.data);
      if (parsed.event === "error") callbacks.onError?.(new Error(String(parsed.data.error ?? "Stream error")));
    }
  }
}

function parseSseFrame(frame: string): { event: string; data: Record<string, unknown> } | null {
  const event = frame
    .split("\n")
    .find((line) => line.startsWith("event:"))
    ?.replace("event:", "")
    .trim();
  const dataLine = frame
    .split("\n")
    .find((line) => line.startsWith("data:"))
    ?.replace("data:", "")
    .trim();

  if (!event || !dataLine) return null;

  try {
    return { event, data: JSON.parse(dataLine) as Record<string, unknown> };
  } catch {
    return { event, data: { delta: dataLine } };
  }
}
