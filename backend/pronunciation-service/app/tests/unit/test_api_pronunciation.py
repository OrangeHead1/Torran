import os
import sys
import pytest
from fastapi.testclient import TestClient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from main import app

client = TestClient(app)

def test_pronunciation_api_predict(monkeypatch):
    class DummyPipeline:
        def predict(self, text, max_len=32):
            return 'aɪˈpiːeɪ'  # Dummy IPA output
    import api.pronunciation as pronunciation_api
    monkeypatch.setattr(pronunciation_api, 'pipeline', DummyPipeline())
    payload = {"text": "IPA"}
    response = client.post("/api/pronunciation/predict-ipa", json=payload)
    assert response.status_code == 200
    assert response.json()["ipa"] == 'aɪˈpiːeɪ'
