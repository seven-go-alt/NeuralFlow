"use client";

export default function DocumentsError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-zinc-950 px-6 text-zinc-100">
      <div className="grid h-14 w-14 place-items-center rounded-xl border border-rose-400/30 bg-rose-400/10">
        <svg className="h-6 w-6 text-rose-300" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M18.364 18.364A9 9 0 0 0 5.636 5.636m12.728 12.728A9 9 0 0 1 5.636 5.636m12.728 12.728L5.636 5.636" />
        </svg>
      </div>
      <h1 className="mt-5 text-xl font-semibold">Failed to load documents</h1>
      <p className="mt-2 text-sm text-zinc-400">{error.message || "The document service is unavailable."}</p>
      <button
        onClick={reset}
        className="mt-6 rounded-lg border border-zinc-700 bg-zinc-900 px-5 py-2 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-800"
      >
        Retry
      </button>
    </main>
  );
}
