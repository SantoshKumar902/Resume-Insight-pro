# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

## Artifacts

### `resume-analyzer` (web, React + Vite)
"Cerebro" — AI-powered resume analyzer. Dashboard SaaS UI in dark theme with sidebar navigation. Pages: Dashboard, Upload Resume, Analysis detail, Compare. Uses Recharts for visualizations, react-hook-form for the upload form, react-dropzone for drag-and-drop, generated React Query hooks from `@workspace/api-client-react`.

### `api-server` (Express)
Backend for the resume analyzer.
- **PDF extraction**: `unpdf`
- **AI**: `@workspace/integrations-gemini-ai` (`gemini-2.5-flash` with structured `responseSchema`)
- **Storage**: `express-session` (in-memory MemoryStore) keyed by sid; resumes are session-scoped
- **Endpoints** (all under `/api`): `GET/DELETE /resumes`, `GET /resumes/summary`, `POST /resumes/analyze` (multipart), `GET/DELETE /resumes/{id}`, `POST /resumes/compare`
- **Required env**: `SESSION_SECRET`, `AI_INTEGRATIONS_GEMINI_BASE_URL`, `AI_INTEGRATIONS_GEMINI_API_KEY`

### `mockup-sandbox` (canvas/design tool, scaffolded but unused)
