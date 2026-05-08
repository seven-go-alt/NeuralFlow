"use client";

import { useMutation, useQuery } from "@tanstack/react-query";

import { Select } from "@/components/ui/select";
import { apiClient } from "@/services/api-client";
import { useAgentStore } from "@/store/agent-store";

export function ModelSelector() {
  const apiBaseUrl = useAgentStore((state) => state.apiBaseUrl);
  const model = useAgentStore((state) => state.model);
  const setModel = useAgentStore((state) => state.setModel);
  const client = apiClient(apiBaseUrl);

  const { data } = useQuery({ queryKey: ["models", apiBaseUrl], queryFn: () => client.models() });
  const switchModel = useMutation({ mutationFn: (nextModel: string) => client.switchModel(nextModel) });
  const models = data?.models.length ? data.models : [model, "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"];

  return (
    <div className="space-y-2">
      <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Model</div>
      <Select
        value={model}
        onChange={(event) => {
          setModel(event.target.value);
          switchModel.mutate(event.target.value);
        }}
      >
        {Array.from(new Set(models)).map((item) => (
          <option key={item} value={item}>
            {item}
          </option>
        ))}
      </Select>
      {data?.error && <p className="text-[11px] leading-4 text-amber-300">{data.error}</p>}
    </div>
  );
}
