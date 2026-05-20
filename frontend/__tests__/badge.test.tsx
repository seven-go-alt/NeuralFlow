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

  it("renders with all tone variants without error", () => {
    const tones = ["cyan", "emerald", "amber", "rose", "violet", "zinc"] as const;
    for (const tone of tones) {
      const { container } = render(<Badge tone={tone}>{tone}</Badge>);
      expect(container.firstChild).toBeInTheDocument();
    }
  });

  it("applies pulse class when pulse prop is true", () => {
    render(<Badge pulse>live</Badge>);
    const el = screen.getByText("live");
    expect(el.className).toContain("animate-pulse-glow");
  });

  it("merges custom className with default classes", () => {
    render(<Badge className="my-custom-class">test</Badge>);
    const el = screen.getByText("test");
    expect(el.className).toContain("my-custom-class");
  });
});
