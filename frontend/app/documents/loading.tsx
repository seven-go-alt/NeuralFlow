export default function Loading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-950">
      <div className="flex flex-col items-center gap-3">
        <div className="h-7 w-7 animate-spin rounded-full border-2 border-zinc-700 border-t-cyan-400" />
        <p className="text-sm text-zinc-500">Loading documents&hellip;</p>
      </div>
    </div>
  );
}
