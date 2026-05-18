"use client";

import { Upload, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

import { Button } from "@/components/ui/button";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? (typeof window !== "undefined" ? `${window.location.protocol}//${window.location.hostname}:20004` : "http://localhost:8000");

export function DocumentUpload({ onUploaded }: { onUploaded?: () => void }) {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  async function onUpload(fileOverride?: File) {
    const file = fileOverride ?? selectedFile ?? inputRef.current?.files?.[0];
    if (!file) {
      setMessage("Please choose a file first");
      return;
    }
    const form = new FormData();
    form.append("file", file);
    form.append("title", file.name.replace(/\.[^.]+$/, ""));
    setUploading(true);
    setMessage("");
    try {
      const response = await fetch(`${API_BASE}/api/v1/documents/upload`, { method: "POST", body: form });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || "Upload failed");
      }
      const data = await response.json();
      setMessage(`Uploaded: ${data.document_id} (${data.status})`);
      setSelectedFile(null);
      if (inputRef.current) inputRef.current.value = "";
      await onUploaded?.();
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
        <input
          ref={inputRef}
          id="file-input"
          type="file"
          accept=".pdf,.md,.markdown,.txt,.docx"
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0] ?? null;
            setSelectedFile(file);
            setMessage(file ? `Selected: ${file.name}` : "");
          }}
          disabled={uploading}
        />
        <div className="flex min-w-0 flex-1 items-center gap-3 rounded-md border border-white/10 bg-black/20 px-3 py-2 text-sm text-zinc-300">
          <Upload className="h-4 w-4 shrink-0 text-cyan-300" />
          <span className="truncate">{selectedFile?.name ?? "No file selected"}</span>
          {selectedFile ? (
            <button
              type="button"
              className="ml-auto shrink-0 text-zinc-500 hover:text-zinc-200"
              onClick={() => {
                setSelectedFile(null);
                setMessage("");
                if (inputRef.current) inputRef.current.value = "";
              }}
              disabled={uploading}
              aria-label="Clear selected file"
            >
              <X className="h-4 w-4" />
            </button>
          ) : null}
        </div>
        <div className="flex items-center gap-2">
          <label htmlFor="file-input">
            <Button type="button" variant="secondary" disabled={uploading} className="whitespace-nowrap">
              Choose file
            </Button>
          </label>
          <Button
            type="button"
            disabled={uploading || !selectedFile}
            className="whitespace-nowrap"
            onClick={() => void onUpload()}
          >
            {uploading ? "Uploading..." : "Upload"}
          </Button>
        </div>
      </div>
      <div className="mt-2 text-xs text-zinc-500">Supported: PDF / Markdown / TXT / DOCX</div>
      {message && <div className="mt-3 text-xs text-cyan-200">{message}</div>}
    </div>
  );
}
