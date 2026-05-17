"use client";

import { FormEvent, useRef, useState } from "react";
import { Brain, CornerDownLeft, Database, Loader2, Paperclip, Square, Wrench } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

export function ChatComposer({
  isStreaming,
  onSubmit,
  onStop,
}: {
  isStreaming: boolean;
  onSubmit: (message: string) => void;
  onStop: () => void;
}) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement | null>(null);

  function submit(event?: FormEvent) {
    event?.preventDefault();
    const message = value.trim();
    if (!message || isStreaming) return;
    setValue("");
    onSubmit(message);
  }

  return (
    <form onSubmit={submit} className="hairline-panel border-t bg-zinc-950/75 p-3 backdrop-blur md:p-4">
      <div className="mx-auto max-w-5xl">
        <div className="mb-2 hidden items-center gap-2 text-[11px] text-zinc-500 md:flex">
          <span className="inline-flex items-center gap-1 rounded-md border border-violet-400/20 bg-violet-400/10 px-2 py-1 text-violet-200">
            <Brain className="h-3 w-3" />
            reasoning
          </span>
          <span className="inline-flex items-center gap-1 rounded-md border border-emerald-400/20 bg-emerald-400/10 px-2 py-1 text-emerald-200">
            <Database className="h-3 w-3" />
            retrieval
          </span>
          <span className="inline-flex items-center gap-1 rounded-md border border-amber-400/20 bg-amber-400/10 px-2 py-1 text-amber-200">
            <Wrench className="h-3 w-3" />
            tools
          </span>
        </div>
        <div className="flex gap-2">
        <Button type="button" variant="outline" size="icon" title="Attach context (coming soon)" disabled>
          <Paperclip className="h-4 w-4" />
        </Button>
        <div className="relative flex-1">
          <Textarea
            ref={ref}
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if ((event.metaKey || event.ctrlKey) && event.key === "Enter") submit();
            }}
            placeholder="Ask the runtime to retrieve memory, call tools, inspect code, or reason through a workflow..."
            className="max-h-40 min-h-11 pr-24"
          />
          <div className="absolute bottom-2 right-2 flex items-center gap-2">
            <span className="hidden text-[11px] text-zinc-500 sm:block">Cmd Enter</span>
            {isStreaming ? (
              <Button type="button" size="sm" variant="destructive" onClick={onStop}>
                <Square className="h-3.5 w-3.5" />
                Stop
              </Button>
            ) : (
              <Button type="submit" size="sm" disabled={!value.trim()}>
                {isStreaming ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <CornerDownLeft className="h-3.5 w-3.5" />}
                Run
              </Button>
            )}
          </div>
        </div>
        </div>
      </div>
    </form>
  );
}
