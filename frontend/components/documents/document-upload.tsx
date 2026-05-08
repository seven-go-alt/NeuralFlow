"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

import { Button } from "@/components/ui/button";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export function DocumentUpload({ onUploaded }: { onUploaded?: () => void }) {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState<string>("");

  async function onUpload() {
    const file = inputRef.current?.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    form.append("title", file.name.replace(/\.[^.]+$/, ""));
    setUploading(true);
    setMessage("");
    try {
      const response = await fetch(`${API_BASE}/api/documents/upload`, { method: "POST", body: form });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || "Upload failed");
      }
      const data = await response.json();
      setMessage(`Uploaded: ${data.document_id} (${data.status})`);
      inputRef.current!.value = "";
      onUploaded?.();
      router.refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="rounded-xl border bg-zinc-950/50 p-4">
      <div className="mb-3 text-sm font-medium text-zinc-100">Upload knowledge documents</div>
      <div className="flex flex-col gap-3 md:flex-row md:items-center">
        <input ref={inputRef} type="file" accept=".pdf,.md,.markdown,.txt,.docx" className="block w-full text-sm text-zinc-400" />
        <Button onClick={onUpload} disabled={uploading}>{uploading ? "Uploading..." : "Upload"}</Button>
      </div>
      <div className="mt-2 text-xs text-zinc-500">Supported: PDF / Markdown / TXT / DOCX</div>
      {message && <div className="mt-3 text-xs text-cyan-200">{message}</div>}
    </div>
  );
}
