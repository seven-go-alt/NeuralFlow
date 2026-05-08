# NeuralFlow Console Frontend

Production-grade AI agent runtime console built with Next.js 15 App Router, TypeScript, TailwindCSS, shadcn-style primitives, Zustand, TanStack Query, and SSE streaming.

## Architecture

```text
frontend/
├── app/                    # App Router entry, providers, global theme
├── components/ui/          # shadcn-style reusable primitives
├── features/
│   ├── chat/               # composer, message list, markdown/code rendering
│   ├── layout/             # app shell, sidebar, responsive chrome
│   ├── runtime/            # pipeline, metrics, chunks, tool calls
│   └── sidebar/            # sessions, model selector, resources
├── hooks/                  # keyboard shortcuts, auto-scroll
├── lib/                    # formatting and class utilities
├── services/               # API abstraction and SSE parser
├── store/                  # Zustand console state
└── types/                  # Agent runtime domain types
```

## State Strategy

Zustand owns local console state: sessions, active messages, selected model, execution mode, in-flight streaming status, and the runtime snapshot. TanStack Query owns remote server state: health, models, skills, and future admin/runtime config reads.

Optimistic chat updates add the user message and an empty assistant message immediately, then stream deltas into the assistant bubble while runtime events update the right panel.

## API Strategy

`services/api-client.ts` wraps FastAPI endpoints:

- `GET /healthz`
- `GET /api/models`
- `POST /api/models/switch`
- `GET /api/skills`
- `POST /chat/react`
- `POST /chat/orchestrate`

`services/streaming.ts` posts to `/chat/stream?include_thinking=true` and parses SSE frames for `message`, `thinking`, `done`, and `error` events.

Set `NEXT_PUBLIC_API_BASE_URL` to point the console at another backend. It defaults to `http://localhost:8000`.

## Runtime Visualization

The right panel intentionally exposes the agent internals rather than hiding them behind a chatbot surface:

- Thinking and routing state
- RAG retrieval status and retrieved chunks
- Function calls and MCP execution output
- Memory and compression phases
- Latency and token metrics

Backend events that are not yet streamed as first-class phases are represented as optimistic runtime events, then hydrated from ReAct/orchestration responses when those modes are used.

## Design System

The UI uses a restrained dark console aesthetic inspired by OpenAI Playground, Linear, Vercel, RAGFlow, and LangGraph Studio:

- 8px-radius panels and controls
- Zinc base surfaces with cyan, emerald, amber, violet, and rose semantic accents
- Dense dashboard layout optimized for scanning repeated runs
- Icon-first controls with lucide-react
- Markdown and syntax-highlighted code output
- Responsive layout: sidebar collapses on tablet, runtime panel hides below desktop

## Commands

```bash
npm install
npm run dev
npm run lint
npm run typecheck
npm run build
```
