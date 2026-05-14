"use client";

import dynamic from "next/dynamic";

const AppShell = dynamic(() => import("@/features/layout/app-shell").then((mod) => mod.AppShell), {
  ssr: false,
  loading: () => <main className="flex h-screen items-center justify-center text-zinc-100 bg-zinc-950">Loading NeuralFlow…</main>,
});

export function AppShellClient() {
  return <AppShell />;
}
