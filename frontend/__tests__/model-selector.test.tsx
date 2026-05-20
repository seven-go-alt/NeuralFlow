import { render, screen, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import React from "react";

// Mock the API client to prevent actual network requests
vi.mock("@/services/api-client", () => ({
  apiClient: () => ({
    models: vi.fn().mockResolvedValue({ models: [] }),
    switchModel: vi.fn(),
  }),
}));

import { ModelSelector } from "@/features/sidebar/model-selector";
import { useAgentStore } from "@/store/agent-store";

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return React.createElement(QueryClientProvider, { client: queryClient }, children);
  };
}

describe("ModelSelector", () => {
  beforeEach(() => {
    useAgentStore.setState({
      model: "gpt-5.4",
      apiBaseUrl: "http://localhost:8000",
    });
  });

  it("renders the model label", () => {
    render(<ModelSelector />, { wrapper: createWrapper() });
    expect(screen.getByText("Model")).toBeInTheDocument();
  });

  it("renders a select element", () => {
    render(<ModelSelector />, { wrapper: createWrapper() });
    const select = screen.getByRole("combobox");
    expect(select).toBeInTheDocument();
  });

  it("calls setModel when a different option is selected", () => {
    const setModelSpy = vi.spyOn(useAgentStore.getState(), "setModel");
    render(<ModelSelector />, { wrapper: createWrapper() });

    const select = screen.getByRole("combobox");
    fireEvent.change(select, { target: { value: "gpt-5.4-mini" } });
    expect(setModelSpy).toHaveBeenCalledWith("gpt-5.4-mini");
    setModelSpy.mockRestore();
  });

  it("displays the current model as the selected value", () => {
    render(<ModelSelector />, { wrapper: createWrapper() });
    const select = screen.getByRole("combobox") as HTMLSelectElement;
    expect(select.value).toBe("gpt-5.4");
  });
});
