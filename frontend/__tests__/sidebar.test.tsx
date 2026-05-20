import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import React from "react";

// Mock child feature components to avoid their dependencies
vi.mock("@/features/sidebar/model-selector", () => ({
  ModelSelector: () => React.createElement("div", { "data-testid": "model-selector" }),
}));

vi.mock("@/features/sidebar/session-list", () => ({
  SessionList: () => React.createElement("div", { "data-testid": "session-list" }),
}));

// Mock Next.js Link to a plain anchor
vi.mock("next/link", () => ({
  default: ({ children, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) =>
    React.createElement("a", props, children),
}));

import { Sidebar } from "@/features/layout/sidebar";
import { useAgentStore } from "@/store/agent-store";

describe("Sidebar", () => {
  it("renders the app title and subtitle", () => {
    render(<Sidebar />);
    expect(screen.getByText("NeuralFlow")).toBeInTheDocument();
    expect(screen.getByText("Agent Runtime Platform")).toBeInTheDocument();
  });

  it("renders the live badge", () => {
    render(<Sidebar />);
    expect(screen.getByText("live")).toBeInTheDocument();
  });

  it("renders child components", () => {
    render(<Sidebar />);
    expect(screen.getByTestId("model-selector")).toBeInTheDocument();
    expect(screen.getByTestId("session-list")).toBeInTheDocument();
  });

  it("renders all mode selection buttons", () => {
    render(<Sidebar />);
    expect(screen.getByText("Streaming")).toBeInTheDocument();
    expect(screen.getByText("ReAct")).toBeInTheDocument();
    expect(screen.getByText("Orchestrate")).toBeInTheDocument();
  });

  it("renders navigation links", () => {
    render(<Sidebar />);
    expect(screen.getByText("Chat")).toBeInTheDocument();
    expect(screen.getByText("Models")).toBeInTheDocument();
    expect(screen.getByText("Status")).toBeInTheDocument();
    expect(screen.getByText("Documents")).toBeInTheDocument();
    expect(screen.getByText("Evaluations")).toBeInTheDocument();
    expect(screen.getByText("Traces")).toBeInTheDocument();
  });

  it("renders the runtime cluster section", () => {
    render(<Sidebar />);
    expect(screen.getByText("Runtime cluster")).toBeInTheDocument();
    expect(screen.getByText("runs")).toBeInTheDocument();
    expect(screen.getByText("tools")).toBeInTheDocument();
    expect(screen.getByText("ctx")).toBeInTheDocument();
  });

  it("highlights the active mode button", () => {
    useAgentStore.setState({ mode: "stream" });
    render(<Sidebar />);
    const streamingBtn = screen.getByText("Streaming").closest("button");
    expect(streamingBtn?.className).toContain("border-cyan-400");
  });

  it("calls setMode when a mode button is clicked", () => {
    useAgentStore.setState({ mode: "stream" });
    const setModeSpy = vi.spyOn(useAgentStore.getState(), "setMode");
    render(<Sidebar />);
    fireEvent.click(screen.getByText("Orchestrate"));
    expect(setModeSpy).toHaveBeenCalledWith("orchestrate");
    setModeSpy.mockRestore();
  });

  it("re-highlights mode after switching active mode", () => {
    useAgentStore.setState({ mode: "react" });
    render(<Sidebar />);
    const reactBtn = screen.getByText("ReAct").closest("button");
    expect(reactBtn?.className).toContain("border-cyan-400");
    // The streaming button should not be highlighted
    const streamingBtn = screen.getByText("Streaming").closest("button");
    expect(streamingBtn?.className).not.toContain("border-cyan-400");
  });
});
