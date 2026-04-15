# Stage 1: Build — install native deps
FROM node:22-bookworm AS builder

RUN apt-get update && apt-get install -y cmake build-essential python3 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev

# Strip CUDA/Vulkan binaries (CPU-only)
RUN rm -rf node_modules/@node-llama-cpp/linux-x64-cuda* \
    && rm -rf node_modules/@node-llama-cpp/linux-x64-vulkan*

# Stage 2: Runtime — slim image
FROM node:22-bookworm-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY package.json server.mjs ./

RUN mkdir -p /home/user/.cache && chown -R node:node /app /home/user

ENV HOME=/home/user

USER node
EXPOSE 8000

CMD ["node", "/app/server.mjs"]
