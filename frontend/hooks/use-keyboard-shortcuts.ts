"use client";

import { useEffect } from "react";

import { useAgentStore } from "@/store/agent-store";

export function useKeyboardShortcuts(onSubmit: () => void) {
  const createSession = useAgentStore((state) => state.createSession);
  const toggleRightPanel = useAgentStore((state) => state.toggleRightPanel);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const mod = event.metaKey || event.ctrlKey;
      if (mod && event.key.toLowerCase() === "enter") {
        event.preventDefault();
        onSubmit();
      }
      if (mod && event.key.toLowerCase() === "k") {
        event.preventDefault();
        createSession();
      }
      if (mod && event.key.toLowerCase() === "j") {
        event.preventDefault();
        toggleRightPanel();
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [createSession, onSubmit, toggleRightPanel]);
}
