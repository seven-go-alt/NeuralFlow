"use client";

import { useCallback, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { Boxes, Database, FileText, GitBranch, Menu, PanelRightOpen, Search, Server, Sparkles, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ChatComposer } from "@/features/chat/chat-composer";
import { MessageList } from "@/features/chat/message-list";
import { RuntimePanel } from "@/features/runtime/runtime-panel";
import { useKeyboardShortcuts } from "@/hooks/use-keyboard-shortcuts";
import { createId } from "@/lib/utils";
import { apiClient } from "@/services/api-client";
import { searchRetrieval } from "@/services/retrieval";
import { streamChat } from "@/services/streaming";
import { useAgentStore } from "@/store/agent-store";
import type { ReactAgentResponse, RuntimeEvent, ToolCall } from "@/types/agent";

import { Sidebar } from "./sidebar";

export function AppShell() {
  const abortRef = useRef<AbortController | null>(null);
  const {
    activeSessionId,
    sessions,
    messages,
    mode,
    apiBaseUrl,
    rightPanelOpen,
    isStreaming,
    addMessage,
    updateMessage,
    appendMessageContent,
    addRuntimeEvent,
    addToolCall,
    setRetrievedChunks,
    setMetrics,
    setRuntimeHint,
    resetRuntime,
    setStreaming,
    toggleRightPanel,
    toggleSidebar,
    setSessionDocument,
  } = useAgentStore();

  const activeSession = sessions.find((session) => session.id === activeSessionId) ?? null;
  const activeDocument = activeSession?.activeDocument ?? null;
  const activeDocumentIds = activeDocument ? [activeDocument.documentId] : [];

  const client = apiClient(apiBaseUrl);
  const { data: health, isError } = useQuery({ queryKey: ["health", apiBaseUrl], queryFn: () => client.health(), refetchInterval: 15_000 });
  const currentMessages = messages[activeSessionId] ?? [];

  const submitMessage = useCallback(
    async (content?: string) => {
      const message = content?.trim();
      if (!message || isStreaming) return;

      const startedAt = performance.now();
      const assistantId = createId();
      let streamedAssistantText = "";
      abortRef.current = new AbortController();
      resetRuntime();
      setStreaming(true);

      addMessage(activeSessionId, {
        id: createId(),
        role: "user",
        content: message,
        createdAt: Date.now(),
        tokens: estimateTokens(message),
      });
      addMessage(activeSessionId, {
        id: assistantId,
        role: "assistant",
        content: "",
        createdAt: Date.now(),
        status: "running",
      });

      seedRuntime(message);
      try {
        const retrieval = await searchRetrieval(message, { documentIds: activeDocumentIds }).catch(() => null);
        if (retrieval?.results?.length) {
          addRuntimeEvent({
            id: createId(),
            type: "retrieval",
            title: activeDocument ? `Document retrieval completed` : "Knowledge retrieval completed",
            detail: `${retrieval.results.length} chunks matched current query.${activeDocument ? ` Scoped to ${activeDocument.title}.` : ""}`,
            status: "success",
            timestamp: Date.now(),
          });
          setRetrievedChunks(
            retrieval.results.map((item) => ({
              id: item.chunk_id,
              source: item.source.filename || item.document_id,
              score: item.score,
              text: item.content,
              documentId: item.document_id,
              chunkId: item.chunk_id,
              pageNumber: item.source.page_number,
              title: item.source.title || item.source.filename || item.document_id,
            })),
          );
          setRuntimeHint(null);
        } else if (activeDocument) {
          setRuntimeHint({
            kind: "warning",
            title: "No matching chunks in current document",
            detail: `Nothing in ${activeDocument.title} matched this query yet. Try a more specific question or reindex the document if it was recently uploaded.`,
          });
        } else {
          setRuntimeHint({
            kind: "info",
            title: "No retrieval evidence yet",
            detail: "The current query did not match any indexed chunks. Try a more specific question or upload a document first.",
          });
        }
        if (mode === "stream") {
          await streamChat({
            baseUrl: apiBaseUrl,
            sessionId: activeSessionId,
            message,
            signal: abortRef.current.signal,
            callbacks: {
              onEvent: (event, data) => {
                if (event === "thinking") runtimeEvent("thinking", "Streaming reasoning", String(data.delta ?? ""), "running");
              },
              onDelta: (delta) => {
                streamedAssistantText += delta;
                appendMessageContent(activeSessionId, assistantId, delta);
              },
              onThinking: (delta) => runtimeEvent("thinking", "Thinking state", delta, "running"),
              onRetrieval: (data) => {
                runtimeEvent("retrieval", activeDocument ? "Document retrieval completed" : "Knowledge retrieval completed", `${String(data.count ?? 0)} chunks matched current query.${activeDocument ? ` Scoped to ${activeDocument.title}.` : ""}`, "success");
              },
              onChunk: (data) => {
                const source = (data.source ?? {}) as Record<string, unknown>;
                setRetrievedChunks([
                  ...useAgentStore.getState().runtime.retrievedChunks,
                  {
                    id: String(data.chunk_id ?? createId()),
                    source: String(source.filename ?? data.document_id ?? "document"),
                    score: Number(data.score ?? 0),
                    text: String(data.content ?? ""),
                    documentId: String(data.document_id ?? ""),
                    chunkId: String(data.chunk_id ?? ""),
                    pageNumber: source.page_number ? Number(source.page_number) : null,
                    title: String(source.title ?? source.filename ?? data.document_id ?? "document"),
                  },
                ]);
                setRuntimeHint(null);
              },
              onDone: (data) => {
                runtimeEvent("metrics", "Stream completed", "SSE response closed successfully.", "success");
                setMetrics({ latencyMs: Number(data.stream_latency ?? 0) * 1000 });
              },
              onError: (error) => runtimeEvent("error", "Stream error", error.message, "error"),
            },
            retrievalOptions: activeDocumentIds.length ? { filters: { document_ids: activeDocumentIds } } : undefined,
          });
          updateMessage(activeSessionId, assistantId, {
            status: "success",
            latencyMs: performance.now() - startedAt,
            tokens: estimateTokens(streamedAssistantText),
          });
        } else {
          const response = await client.chat(activeSessionId, message, { documentIds: activeDocumentIds });
          updateMessage(activeSessionId, assistantId, {
            content: response.reply,
            status: "success",
            latencyMs: performance.now() - startedAt,
            intent: response.intent,
            tokens: estimateTokens(response.reply),
            usedSkills: response.used_skills,
            citations: response.citations,
          });
          if (response.citations?.length) {
            addRuntimeEvent({
              id: createId(),
              type: "retrieval",
              title: "Citations attached",
              detail: `${response.citations.length} sources attached to final answer.${activeDocument ? ` Scoped to ${activeDocument.title}.` : ""}`,
              status: "success",
              timestamp: Date.now(),
            });
            setRuntimeHint(null);
          }
        }
      } catch (error) {
        const err = error instanceof Error ? error : new Error("Request failed");
        updateMessage(activeSessionId, assistantId, { content: err.message, status: "error", latencyMs: performance.now() - startedAt });
        runtimeEvent("error", "Runtime failure", err.message, "error");
        setRuntimeHint({ kind: "error", title: "Request failed", detail: err.message });
      } finally {
        setStreaming(false);
        abortRef.current = null;
      }

      function runtimeEvent(type: RuntimeEvent["type"], title: string, detail: string, status: RuntimeEvent["status"]) {
        addRuntimeEvent({ id: createId(), type, title, detail, status, timestamp: Date.now() });
      }

      function seedRuntime(query: string) {
        addRuntimeEvent({ id: createId(), type: "thinking", title: "Intent router", detail: `Classifying: ${query.slice(0, 90)}`, status: "running", timestamp: Date.now() });
        addRuntimeEvent({
          id: createId(),
          type: "retrieval",
          title: "RAG retrieval queued",
          detail: activeDocument
            ? `Preparing retrieval for ${activeDocument.title}.`
            : "Document knowledge base and memory context are being prepared.",
          status: "pending",
          timestamp: Date.now(),
        });
        setRetrievedChunks([]);
        setRuntimeHint(
          activeDocument
            ? {
                kind: "info",
                title: "Scoped to current document",
                detail: `This session is limited to ${activeDocument.title}. Clear the document badge in the header to search across the full knowledge base.`,
              }
            : null,
        );
      }

      function _hydrateAgentResult(result: ReactAgentResponse) {
        addRuntimeEvent({
          id: createId(),
          type: "thinking",
          title: result.route ? `Routed to ${result.route}` : "ReAct loop complete",
          detail: result.route_reason ?? `${result.total_iterations} iterations executed.`,
          status: "success",
          timestamp: Date.now(),
        });
        result.steps.forEach((step, index) => {
          const toolName = String(step.tool ?? step.name ?? step.action ?? `step_${index + 1}`);
          const type = String(step.type ?? "");
          if (type.includes("tool")) {
            const call: ToolCall = {
              id: createId(),
              name: toolName,
              status: "success",
              input: step.input ?? step.arguments,
              output: step.observation,
            };
            addToolCall(call);
            addRuntimeEvent({ id: createId(), type: "tool_call", title: toolName, detail: JSON.stringify(step.observation ?? step, null, 2).slice(0, 260), status: "success", timestamp: Date.now() });
          } else {
            addRuntimeEvent({ id: createId(), type: "thinking", title: type || `Iteration ${index + 1}`, detail: JSON.stringify(step).slice(0, 260), status: "success", timestamp: Date.now() });
          }
        });
        setMetrics({ latencyMs: performance.now() - startedAt, toolMs: result.total_iterations * 220, tokensOut: estimateTokens(result.final_answer) });
      }
    },
    [
      activeDocument,
      activeDocumentIds,
      activeSessionId,
      addMessage,
      addRuntimeEvent,
      addToolCall,
      apiBaseUrl,
      appendMessageContent,
      client,
      isStreaming,
      mode,
      resetRuntime,
      setMetrics,
      setRetrievedChunks,
      setRuntimeHint,
      setStreaming,
      updateMessage,
    ],
  );

  useKeyboardShortcuts(() => {
    const composer = document.querySelector<HTMLTextAreaElement>("textarea");
    if (composer?.value) submitMessage(composer.value);
  });

  return (
    <main className="console-surface relative flex h-screen text-zinc-100">
      <Sidebar />
      <section className="flex min-w-0 flex-1 flex-col">
        <header className="hairline-panel flex h-16 shrink-0 items-center justify-between border-b bg-zinc-950/70 px-3 backdrop-blur md:px-5">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" className="md:hidden" onClick={toggleSidebar}>
              <Menu className="h-4 w-4" />
            </Button>
            <div className="grid h-9 w-9 place-items-center rounded-lg border border-cyan-300/30 bg-cyan-300/10">
              <Sparkles className="h-4 w-4 text-cyan-200" />
            </div>
            <div>
              <div className="text-sm font-semibold">Agent Console</div>
              <div className="text-[11px] text-zinc-500">Runtime mode: {mode}</div>
            </div>
            {activeDocument ? (
              <div className="hidden items-center gap-2 rounded-lg border border-cyan-400/30 bg-cyan-400/10 px-3 py-2 md:flex">
                <FileText className="h-3.5 w-3.5 text-cyan-200" />
                <div className="max-w-56 truncate text-xs text-cyan-100">{activeDocument.title}</div>
                <button
                  type="button"
                  className="text-zinc-400 transition hover:text-zinc-200"
                  onClick={() => setSessionDocument(activeSessionId, null)}
                  title="Clear document scope"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            ) : null}
          </div>
          <div className="mx-4 hidden min-w-0 max-w-xl flex-1 items-center rounded-lg border bg-zinc-950/70 px-3 py-2 text-xs text-zinc-500 2xl:flex">
            <Search className="mr-2 h-3.5 w-3.5 text-zinc-600" />
            Inspect runs, sessions, tools, memory, or paste a trace id
            <span className="ml-auto rounded border border-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-500">Cmd K</span>
          </div>
          <div className="flex items-center gap-3 text-xs text-zinc-500">
            <div className="hidden items-center gap-2 rounded-lg border bg-zinc-950/60 px-3 py-2 2xl:flex">
              <Database className="h-3.5 w-3.5 text-emerald-300" />
              <span>RAG</span>
              <Boxes className="ml-1 h-3.5 w-3.5 text-amber-300" />
              <span>MCP</span>
              <GitBranch className="ml-1 h-3.5 w-3.5 text-violet-300" />
              <span>Graph</span>
            </div>
            <div className="flex items-center gap-2 rounded-lg border bg-zinc-950/60 px-3 py-2">
              <Server className={`h-3.5 w-3.5 ${health && !isError ? "text-emerald-300" : "text-amber-300"}`} />
              <span className="hidden sm:inline">{health?.app ?? "Backend pending"}</span>
            </div>
            {!rightPanelOpen && (
              <Button variant="outline" size="sm" onClick={toggleRightPanel}>
                <PanelRightOpen className="h-3.5 w-3.5" />
                Runtime
              </Button>
            )}
          </div>
        </header>
        <MessageList messages={currentMessages} onRetry={() => submitMessage(currentMessages.at(-2)?.content)} />
        <ChatComposer isStreaming={isStreaming} onSubmit={submitMessage} onStop={() => abortRef.current?.abort()} />
      </section>
      {rightPanelOpen && <RuntimePanel />}
    </main>
  );
}

function estimateTokens(text: string) {
  return Math.max(1, Math.ceil(text.length / 4));
}
