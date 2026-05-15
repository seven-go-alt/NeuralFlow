# NeuralFlow production deployment with GitHub + GHCR

## Recommended model

- Build images in GitHub Actions
- Push images to GHCR
- On the server, only pull images and start Docker Compose

## Files

- `docker-compose.prod.yml`: production service orchestration using prebuilt images
- `.env.production`: runtime application configuration and secrets
- `.env.images`: image tags to deploy
- `deploy/Caddyfile`: reverse proxy configuration

## Required runtime variables

At minimum, ensure these are set in `.env.production`:

- `POSTGRES_PASSWORD`
- `DATABASE_URL` (must use the same database password as `POSTGRES_PASSWORD`)
- `PUBLIC_BASE_URL`
- `CORS_ALLOW_ORIGINS`
- `LLM_API_BASE` / `LLM_API_KEY` or `OPENAI_API_KEY`
- `EMBEDDING_API_BASE` / `EMBEDDING_API_KEY`
- `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` when outbound model access needs a proxy

## Example `.env.images`

```env
NEURALFLOW_API_IMAGE=ghcr.io/seven-go-alt/neuralflow-api:v0.1.0
NEURALFLOW_FRONTEND_IMAGE=ghcr.io/seven-go-alt/neuralflow-frontend:v0.1.0
```

## Deploy on server

```bash
cd /opt/neuralflow
cp .env.images.example .env.images
# edit .env.images with the tag you want to deploy

docker compose --env-file .env.images -f docker-compose.prod.yml pull
docker compose --env-file .env.images -f docker-compose.prod.yml up -d
docker compose --env-file .env.images -f docker-compose.prod.yml ps
```

## Rollback

Change `.env.images` back to a previous tag, then run:

```bash
docker compose --env-file .env.images -f docker-compose.prod.yml pull
docker compose --env-file .env.images -f docker-compose.prod.yml up -d
```

## Local verification before release

Before pushing release-oriented build changes or cutting a production tag, run local verification first:

```bash
cd /opt/neuralflow
bash scripts/verify-local.sh
```

The script is source-aware for domestic network conditions and defaults to:

- `UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`
- `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`
- `UV_HTTP_TIMEOUT=180`
- `UV_CONCURRENT_DOWNLOADS=1`
- `HTTP_PROXY=http://127.0.0.1:7890`
- `HTTPS_PROXY=http://127.0.0.1:7890`
- `ALL_PROXY=http://127.0.0.1:7890`

Do not push release pipeline changes until local verification passes.

## Build-time proxy notes

When the CI runner or local build host needs a proxy:

- API image build passes `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` into the root `Dockerfile`
- Frontend image build also passes `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` into `frontend/Dockerfile`
- `frontend/.dockerignore` excludes `node_modules` and `.next` to keep Docker build context small

This matters because frontend failures often look like a generic `npm ci` hang, while the actual cause is either:
- missing build args in CI, or
- an oversized Docker context slowing everything to a crawl

If you need details, see `deploy/proxy-guide.md`.

## Recommended validation before tagging

For build-pipeline changes, verify in this order:

1. `cd frontend && npm run lint`
2. `cd frontend && npm run build`
3. Run the local verification script
4. Trigger the GitHub Actions workflow and confirm both API and frontend images publish successfully

## Notes

- Do not rely on `docker compose up --build` on the production host as a long-term workflow.
- Prefer release tags like `v0.1.0` over `latest` for production deployments.
- Expose only 80/443 publicly through Caddy. Keep internal service ports private.
