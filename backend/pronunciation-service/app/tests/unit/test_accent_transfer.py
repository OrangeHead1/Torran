import os
import torch
import pytest
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.accent_transfer import AccentTransferModel, AccentTransferPipeline
from models.load_vocab import load_ipa_vocab

def test_accent_transfer_forward():
    ipa_vocab = load_ipa_vocab(os.path.join(os.path.dirname(__file__), '../../models/ipa_vocab.txt'))
    accent2id = {'american': 0, 'british': 1}
    model = AccentTransferModel(ipa_vocab_size=len(ipa_vocab), accent_count=len(accent2id))
    input_embeds = torch.zeros((1, 5, model.encoder.input_size))
    accent_id = torch.tensor([0])
    decoder_input = torch.zeros((1, 1, model.encoder.input_size))
    logits = model(input_embeds, accent_id, decoder_input)
    assert logits.shape[-1] == len(ipa_vocab)

def test_accent_transfer_pipeline_predict(monkeypatch):
    ipa_vocab = load_ipa_vocab(os.path.join(os.path.dirname(__file__), '../../models/ipa_vocab.txt'))
    accent2id = {'american': 0, 'british': 1}
    class DummyModel(AccentTransferModel):
        def forward(self, input_embeds, accent_id, decoder_input, hidden=None):
            batch, seq, _ = decoder_input.shape
            logits = torch.zeros((batch, seq, len(ipa_vocab)))
            logits[..., 1] = 1  # Always predict <EOS>
            return logits
    monkeypatch.setattr('models.accent_transfer.AccentTransferModel', DummyModel)
    pipeline = AccentTransferPipeline(model_path=None, ipa_vocab=ipa_vocab, accent2id=accent2id)
    pipeline.model = DummyModel(ipa_vocab_size=len(ipa_vocab), accent_count=len(accent2id))
    input_embeds = torch.zeros((1, 5, pipeline.model.encoder.input_size))
    result = pipeline.predict(input_embeds, 'american')
    assert isinstance(result, str)
