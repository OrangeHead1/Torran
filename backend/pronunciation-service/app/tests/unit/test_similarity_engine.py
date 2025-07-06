import os
import torch
import pytest
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.similarity_engine import SimilarityEngine, SimilarityPipeline
from models.load_vocab import load_ipa_vocab

def test_similarity_engine_forward():
    ipa_vocab = load_ipa_vocab(os.path.join(os.path.dirname(__file__), '../../models/ipa_vocab.txt'))
    model = SimilarityEngine(ipa_vocab_size=len(ipa_vocab))
    user_ipa_ids = torch.randint(0, len(ipa_vocab), (1, 5))
    target_ipa_ids = torch.randint(0, len(ipa_vocab), (1, 5))
    score = model(user_ipa_ids, target_ipa_ids)
    assert 0.0 <= score.item() <= 1.0

def test_similarity_pipeline_compare(monkeypatch):
    ipa_vocab = load_ipa_vocab(os.path.join(os.path.dirname(__file__), '../../models/ipa_vocab.txt'))
    class DummyModel(SimilarityEngine):
        def forward(self, user_ipa_ids, target_ipa_ids):
            return torch.tensor([[0.85]])
    monkeypatch.setattr('models.similarity_engine.SimilarityEngine', DummyModel)
    pipeline = SimilarityPipeline(model_path=None, ipa_vocab=ipa_vocab)
    pipeline.model = DummyModel(ipa_vocab_size=len(ipa_vocab))
    # Use only IPA vocab symbols
    ipa_str = ipa_vocab[2] * 3  # e.g., 'aaa' if 'a' is at index 2
    score = pipeline.compare(ipa_str, ipa_str)
    assert abs(score - 0.85) < 1e-5
