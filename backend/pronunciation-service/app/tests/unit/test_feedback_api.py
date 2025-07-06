import os
import sys
import pytest
from fastapi.testclient import TestClient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from main import app

client = TestClient(app)

def test_feedback_api_match():
    payload = {"user_ipa": "a", "target_ipa": "a", "accent": "american"}
    response = client.post("/api/feedback/generate", json=payload)
    assert response.status_code == 200
    assert "matches the target" in response.json()["feedback"]

def test_feedback_api_diff():
    payload = {"user_ipa": "a", "target_ipa": "b", "accent": "american"}
    response = client.post("/api/feedback/generate", json=payload)
    assert response.status_code == 200
    assert "Focus on the differences" in response.json()["feedback"]
