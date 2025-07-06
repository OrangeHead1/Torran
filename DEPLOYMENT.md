# Adaptive Pronunciation Coach: Deployment & Development

## Local Development

- Start all services:
  ```sh
  ./scripts/dev.sh
  ```
- Stop all services:
  ```sh
  ./scripts/stop.sh
  ```

## Docker Compose
- `docker-compose.yml` orchestrates the pronunciation service and API gateway.
- Each service has its own `Dockerfile` for production builds.

## Structure
- `scripts/dev.sh` — Start all services
- `scripts/stop.sh` — Stop all services
- `backend/pronunciation-service/app/Dockerfile` — Pronunciation service image
- `backend/api-gateway/app/Dockerfile` — API gateway image

## Next Steps
- Add persistent storage (Postgres, Redis, etc.)
- Add production-ready security and monitoring
- Deploy to cloud (Kubernetes, AWS, GCP, Azure, etc.)
