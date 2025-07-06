# Adaptive Pronunciation Coach: Backend Pronunciation Service

This service provides advanced, production-grade pronunciation analysis using transformer-based models for IPA prediction, accent transfer, and similarity scoring. It is designed for robust, scalable deployment as part of a microservices architecture.

## Features
- **IPA Prediction**: Transformer-based grapheme-to-phoneme model (BERT encoder + LSTM decoder)
- **Accent Transfer**: Multi-accent transformer for accent adaptation
- **Similarity Engine**: Deep similarity scoring between user and target IPA
- **FastAPI Endpoints**: RESTful APIs for prediction, analysis, and feedback
- **Extensive Unit Tests**: All core logic and endpoints are covered

## Structure
- `models/` — All ML models and vocab
- `api/` — FastAPI endpoints for each service
- `tests/unit/` — Unit tests for all models and APIs
- `main.py` — FastAPI app entrypoint
- `requirements.txt` — Python dependencies

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the service: `uvicorn main:app --reload`
3. Use the API endpoints for IPA prediction, similarity, and feedback

## Model Vocab
- IPA vocab is in `models/ipa_vocab.txt` and must include <PAD>, <SOS>, <EOS> tokens at the end.

## License
This project uses only free/open-source technologies.
