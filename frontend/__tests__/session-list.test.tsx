import { render, screen, fireEvent } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SessionList } from "@/features/sidebar/session-list";
import { useAgentStore } from "@/store/agent-store";

describe("SessionList", () => {
  beforeEach(() => {
    useAgentStore.setState({
      sessions: [
        {
          id: "sess_1",
          title: "Runtime console",
          updatedAt: Date.now(),
          model: "gpt-5.4",
          messageCount: 0,
        },
        {
          id: "sess_2",
          title: "Agent debugging",
          updatedAt: Date.now(),
          model: "gpt-5.4-mini",
          messageCount: 5,
        },
      ],
      activeSessionId: "sess_1",
    });
  });

  it("renders all session titles", () => {
    render(<SessionList />);
    expect(screen.getByText("Runtime console")).toBeInTheDocument();
    expect(screen.getByText("Agent debugging")).toBeInTheDocument();
  });

  it("renders message count for each session", () => {
    render(<SessionList />);
    expect(screen.getByText("0 messages")).toBeInTheDocument();
    expect(screen.getByText("5 messages")).toBeInTheDocument();
  });

  it("renders model name for each session", () => {
    render(<SessionList />);
    expect(screen.getByText("gpt-5.4")).toBeInTheDocument();
    expect(screen.getByText("gpt-5.4-mini")).toBeInTheDocument();
  });

  it("highlights the active session with cyan border", () => {
    render(<SessionList />);
    const activeBtn = screen.getByText("Runtime console").closest("button");
    expect(activeBtn?.className).toContain("border-cyan-400");
  });

  it("does not apply cyan border to inactive sessions", () => {
    render(<SessionList />);
    const inactiveBtn = screen.getByText("Agent debugging").closest("button");
    expect(inactiveBtn?.className).not.toContain("border-cyan-400");
  });

  it("calls setActiveSession when a session is clicked", () => {
    const setActiveSpy = vi.spyOn(useAgentStore.getState(), "setActiveSession");
    render(<SessionList />);
    fireEvent.click(screen.getByText("Agent debugging"));
    expect(setActiveSpy).toHaveBeenCalledWith("sess_2");
    setActiveSpy.mockRestore();
  });

  it("calls createSession when the new session button is clicked", () => {
    const createSpy = vi.spyOn(useAgentStore.getState(), "createSession");
    render(<SessionList />);
    const newBtn = screen.getByTitle("New session");
    expect(newBtn).toBeInTheDocument();
    fireEvent.click(newBtn);
    expect(createSpy).toHaveBeenCalled();
    createSpy.mockRestore();
  });
});
