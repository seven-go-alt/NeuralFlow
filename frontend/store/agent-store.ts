"use client";

import { create } from "zustand";

import { createId, createSessionId } from "@/lib/utils";
import type {
  ActiveDocumentContext,
  ChatMessage,
  ChatMode,
  ConversationSession,
  RuntimeEvent,
  RuntimeHint,
  RuntimeMetrics,
  RuntimeSnapshot,
  ToolCall,
  RetrievedChunk,
} from "@/types/agent";

const initialSessionId = createSessionId();

interface AgentState {
  activeSessionId: string;
  sessions: ConversationSession[];
  messages: Record<string, ChatMessage[]>;
  runtime: RuntimeSnapshot;
  mode: ChatMode;
  model: string;
  apiBaseUrl: string;
  rightPanelOpen: boolean;
  sidebarOpen: boolean;
  isStreaming: boolean;
  setMode: (mode: ChatMode) => void;
  setModel: (model: string) => void;
  setApiBaseUrl: (url: string) => void;
  setActiveSession: (id: string) => void;
  createSession: () => string;
  createSessionWithDocument: (document: ActiveDocumentContext, options?: { initialPrompt?: string }) => string;
  setSessionDocument: (sessionId: string, document: ActiveDocumentContext | null) => void;
  clearPendingPrompt: (sessionId: string) => void;
  addMessage: (sessionId: string, message: ChatMessage) => void;
  updateMessage: (sessionId: string, messageId: string, patch: Partial<ChatMessage>) => void;
  appendMessageContent: (sessionId: string, messageId: string, delta: string) => void;
  addRuntimeEvent: (event: RuntimeEvent) => void;
  setRuntimeEvents: (events: RuntimeEvent[]) => void;
  addToolCall: (toolCall: ToolCall) => void;
  setRetrievedChunks: (chunks: RetrievedChunk[]) => void;
  setMetrics: (metrics: RuntimeMetrics) => void;
  setRuntimeHint: (hint: RuntimeHint | null) => void;
  resetRuntime: () => void;
  setStreaming: (value: boolean) => void;
  toggleRightPanel: () => void;
  toggleSidebar: () => void;
}

const emptyRuntime = (): RuntimeSnapshot => ({
  events: [],
  retrievedChunks: [],
  toolCalls: [],
  metrics: {},
  hint: null,
});

export const useAgentStore = create<AgentState>((set, get) => ({
  activeSessionId: initialSessionId,
  sessions: [
    {
      id: initialSessionId,
      title: "Runtime console",
      updatedAt: Date.now(),
      model: "gpt-5.4",
      messageCount: 0,
      activeDocument: null,
    },
  ],
  messages: { [initialSessionId]: [] },
  runtime: emptyRuntime(),
  mode: "stream",
  model: "gpt-5.4",
  apiBaseUrl: process.env.NEXT_PUBLIC_API_BASE_URL ?? (typeof window !== "undefined" ? `${window.location.protocol}//${window.location.hostname}:20004` : "http://localhost:8000"),
  rightPanelOpen: true,
  sidebarOpen: false,
  isStreaming: false,
  setMode: (mode) => set({ mode }),
  setModel: (model) => set({ model }),
  setApiBaseUrl: (apiBaseUrl) => set({ apiBaseUrl }),
  setActiveSession: (activeSessionId) => set({ activeSessionId, runtime: emptyRuntime() }),
  createSession: () => {
    const id = createSessionId();
    set((state) => ({
      activeSessionId: id,
      runtime: emptyRuntime(),
      sessions: [
        {
          id,
          title: "New agent run",
          updatedAt: Date.now(),
          model: state.model,
          messageCount: 0,
          activeDocument: null,
        },
        ...state.sessions,
      ],
      messages: { ...state.messages, [id]: [] },
    }));
    return id;
  },
  createSessionWithDocument: (document, options) => {
    const id = createSessionId();
    const initialPrompt = options?.initialPrompt?.trim();
    set((state) => ({
      activeSessionId: id,
      runtime: emptyRuntime(),
      sessions: [
        {
          id,
          title: document.title || "Document chat",
          updatedAt: Date.now(),
          model: state.model,
          messageCount: 0,
          activeDocument: document,
          pendingPrompt: initialPrompt || undefined,
        },
        ...state.sessions,
      ],
      messages: { ...state.messages, [id]: [] },
    }));
    return id;
  },
  setSessionDocument: (sessionId, document) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.id === sessionId
          ? {
              ...session,
              activeDocument: document,
              updatedAt: Date.now(),
            }
          : session,
      ),
    })),
  clearPendingPrompt: (sessionId) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.id === sessionId
          ? { ...session, pendingPrompt: undefined }
          : session,
      ),
    })),
  addMessage: (sessionId, message) =>
    set((state) => ({
      messages: {
        ...state.messages,
        [sessionId]: [...(state.messages[sessionId] ?? []), message],
      },
      sessions: state.sessions.map((session) =>
        session.id === sessionId
          ? {
              ...session,
              title: message.role === "user" ? message.content.slice(0, 42) || session.title : session.title,
              updatedAt: Date.now(),
              messageCount: (state.messages[sessionId]?.length ?? 0) + 1,
            }
          : session,
      ),
    })),
  updateMessage: (sessionId, messageId, patch) =>
    set((state) => ({
      messages: {
        ...state.messages,
        [sessionId]: (state.messages[sessionId] ?? []).map((message) =>
          message.id === messageId ? { ...message, ...patch } : message,
        ),
      },
    })),
  appendMessageContent: (sessionId, messageId, delta) =>
    get().updateMessage(sessionId, messageId, {
      content: `${(get().messages[sessionId] ?? []).find((message) => message.id === messageId)?.content ?? ""}${delta}`,
    }),
  addRuntimeEvent: (event) =>
    set((state) => ({
      runtime: { ...state.runtime, events: [event, ...state.runtime.events].slice(0, 40) },
    })),
  setRuntimeEvents: (events) => set((state) => ({ runtime: { ...state.runtime, events } })),
  addToolCall: (toolCall) =>
    set((state) => ({
      runtime: { ...state.runtime, toolCalls: [toolCall, ...state.runtime.toolCalls].slice(0, 16) },
    })),
  setRetrievedChunks: (retrievedChunks) => set((state) => ({ runtime: { ...state.runtime, retrievedChunks } })),
  setMetrics: (metrics) => set((state) => ({ runtime: { ...state.runtime, metrics: { ...state.runtime.metrics, ...metrics } } })),
  setRuntimeHint: (hint) => set((state) => ({ runtime: { ...state.runtime, hint } })),
  resetRuntime: () => set({ runtime: emptyRuntime() }),
  setStreaming: (isStreaming) => set({ isStreaming }),
  toggleRightPanel: () => set((state) => ({ rightPanelOpen: !state.rightPanelOpen })),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
}));
