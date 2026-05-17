import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RetrievedChunks } from "@/features/runtime/retrieved-chunks";
import type { RetrievedChunk } from "@/types/agent";

describe("RetrievedChunks", () => {
  it("shows empty state when no chunks or hint", () => {
    render(<RetrievedChunks chunks={[]} />);
    expect(screen.getByText(/RAG evidence will appear/)).toBeInTheDocument();
  });

  it("renders chunks with scores", () => {
    const chunks: RetrievedChunk[] = [
      {
        id: "chunk_1",
        source: "handbook.pdf",
        score: 0.91,
        text: "员工请假制度",
        documentId: "doc_1",
        chunkId: "chk_1",
      },
    ];
    render(<RetrievedChunks chunks={chunks} />);
    expect(screen.getByText("91%")).toBeInTheDocument();
    expect(screen.getByText("员工请假制度")).toBeInTheDocument();
  });
});
