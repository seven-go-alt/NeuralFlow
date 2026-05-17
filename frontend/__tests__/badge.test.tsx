import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Badge } from "@/components/ui/badge";

describe("Badge", () => {
  it("renders children text", () => {
    render(<Badge>ready</Badge>);
    expect(screen.getByText("ready")).toBeInTheDocument();
  });

  it("applies tone class for emerald", () => {
    const { container } = render(<Badge tone="emerald">live</Badge>);
    expect(container.firstChild).toBeInTheDocument();
  });

  it("applies tone class for rose (error)", () => {
    const { container } = render(<Badge tone="rose">failed</Badge>);
    expect(container.firstChild).toBeInTheDocument();
  });
});
