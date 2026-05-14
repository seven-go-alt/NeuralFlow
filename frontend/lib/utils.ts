import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatLatency(ms?: number) {
  if (ms === undefined) return "--";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function compactNumber(value?: number) {
  if (value === undefined) return "--";
  return Intl.NumberFormat("en", { notation: "compact" }).format(value);
}

function randomId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

export function createSessionId() {
  return `sess_${randomId().slice(0, 8)}`;
}

export function createId() {
  return randomId();
}
