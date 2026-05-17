import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MessageBubble } from "@/features/chat/message-bubble";
import type { ChatMessage } from "@/types/agent";

describe("MessageBubble", () => {
  const userMessage: ChatMessage = {
    id: "1",
    role: "user",
    content: "员工请假制度是什么？",
    createdAt: Date.now(),
  };

  const assistantMessage: ChatMessage = {
    id: "2",
    role: "assistant",
    content: "根据公司规定，员工每年享有5天年假。",
    createdAt: Date.now(),
    intent: "general",
    tokens: 18,
    citations: [
      { index: 1, label: "Employee Handbook", document_id: "doc_1", chunk_id: "chk_1", page_number: 3 },
    ],
  };

  it("renders user message", () => {
    render(<MessageBubble message={userMessage} />);
    expect(screen.getByText("员工请假制度是什么？")).toBeInTheDocument();
  });

  it("renders assistant response with intent badge", () => {
    render(<MessageBubble message={assistantMessage} />);
    expect(screen.getByText("根据公司规定，员工每年享有5天年假。")).toBeInTheDocument();
    expect(screen.getByText("general")).toBeInTheDocument();
  });

  it("shows token count", () => {
    render(<MessageBubble message={assistantMessage} />);
    expect(screen.getByText("18 tokens")).toBeInTheDocument();
  });

  it("shows loading state when content is empty", () => {
    const loading: ChatMessage = { id: "3", role: "assistant", content: "", createdAt: Date.now(), status: "running" };
    render(<MessageBubble message={loading} />);
    expect(screen.getByText("Waiting for runtime output")).toBeInTheDocument();
  });
});
