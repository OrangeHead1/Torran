# Adaptive Pronunciation Coach: API Gateway

This API Gateway proxies and orchestrates requests to backend microservices, including pronunciation analysis, user management, progress tracking, and more. It is designed for secure, scalable, and maintainable integration in a microservices architecture.

## Features
- **Pronunciation Proxy**: Forwards requests to the pronunciation microservice
- **Extensible Endpoints**: Ready for auth, users, exercises, progress, subscriptions, social, admin, etc.
- **FastAPI**: Modern, async Python API gateway
- **Production-Ready**: Robust error handling and dependency management

## Structure
- `app/api/v1/endpoints/` — All API v1 endpoints
- `app/main.py` — FastAPI app entrypoint
- `app/requirements.txt` — Python dependencies

## Usage
1. Install dependencies: `pip install -r app/requirements.txt`
2. Run the gateway: `uvicorn app.main:app --reload`
3. Use `/api/pronunciation/predict-ipa` and other endpoints

## License
This project uses only free/open-source technologies.
