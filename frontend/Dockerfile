FROM node:22-alpine AS deps
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV HTTP_PROXY=${HTTP_PROXY:-} \
    HTTPS_PROXY=${HTTPS_PROXY:-} \
    NO_PROXY=${NO_PROXY:-}
WORKDIR /app
COPY package.json package-lock.json ./
RUN if [ -n "$HTTP_PROXY" ]; then npm config set proxy "$HTTP_PROXY"; fi \
    && if [ -n "$HTTPS_PROXY" ]; then npm config set https-proxy "$HTTPS_PROXY"; fi \
    && if [ -n "$NO_PROXY" ]; then npm config set noproxy "$NO_PROXY"; fi \
    && npm ci

FROM node:22-alpine AS builder
ARG NEXT_PUBLIC_API_BASE_URL
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL:-http://localhost:8000}
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN mkdir -p public && npm run build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/package-lock.json ./package-lock.json
COPY --from=builder /app/node_modules ./node_modules
EXPOSE 3000
CMD ["npm", "run", "start"]
