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

## Notes

- Do not rely on `docker compose up --build` on the production host as a long-term workflow.
- Prefer release tags like `v0.1.0` over `latest` for production deployments.
- Expose only 80/443 publicly through Caddy. Keep internal service ports private.
