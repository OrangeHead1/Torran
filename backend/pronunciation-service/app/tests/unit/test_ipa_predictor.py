import os
import torch
import pytest
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.ipa_predictor import IPAPredictor, IPAPredictorPipeline
from models.load_vocab import load_ipa_vocab

def test_ipa_predictor_forward():
    ipa_vocab = load_ipa_vocab(os.path.join(os.path.dirname(__file__), '../../models/ipa_vocab.txt'))
    model = IPAPredictor(ipa_vocab_size=len(ipa_vocab))
    input_ids = torch.randint(0, 100, (1, 5))
    attention_mask = torch.ones((1, 5))
    # Use integer indices for decoder_input_ids (e.g., <SOS> token index 0)
    decoder_input_ids = torch.zeros((1, 1), dtype=torch.long)
    logits, hidden = model(input_ids, attention_mask, decoder_input_ids)
    assert logits.shape[-1] == len(ipa_vocab)

def test_pipeline_predict(monkeypatch):
    ipa_vocab = load_ipa_vocab(os.path.join(os.path.dirname(__file__), '../../models/ipa_vocab.txt'))
    # Patch model loading to avoid missing weights
    class DummyModel(IPAPredictor):
        def forward(self, input_ids, attention_mask, decoder_input_ids, hidden=None):
            batch, seq = decoder_input_ids.shape
            logits = torch.zeros((batch, seq, len(ipa_vocab)))
            logits[..., 1] = 1  # Always predict <EOS>
            return logits, hidden
    monkeypatch.setattr('models.ipa_predictor.IPAPredictor', DummyModel)
    pipeline = IPAPredictorPipeline(model_path='dummy.pt', ipa_vocab=ipa_vocab)
    pipeline.model = DummyModel(ipa_vocab_size=len(ipa_vocab))
    result = pipeline.predict("test")
    assert isinstance(result, str)
