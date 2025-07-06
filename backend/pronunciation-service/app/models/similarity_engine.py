import torch
import torch.nn as nn
from typing import List
import os

class SimilarityEngine(nn.Module):
    """
    Neural model for phoneme-level pronunciation similarity scoring.
    """
    def __init__(self, ipa_vocab_size: int, hidden_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(ipa_vocab_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.similarity_fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ipa_ids, target_ipa_ids):
        user_emb = self.embedding(user_ipa_ids)
        target_emb = self.embedding(target_ipa_ids)
        user_out, _ = self.encoder(user_emb)
        target_out, _ = self.encoder(target_emb)
        # Use last hidden state for both
        user_vec = user_out[:, -1, :]
        target_vec = target_out[:, -1, :]
        diff = torch.abs(user_vec - target_vec)
        score = self.similarity_fc(diff)
        return self.sigmoid(score)

class SimilarityPipeline:
    def __init__(self, model_path: str, ipa_vocab: List[str]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ipa_token2id = {t: i for i, t in enumerate(ipa_vocab)}
        self.model = SimilarityEngine(ipa_vocab_size=len(ipa_vocab))
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def compare(self, user_ipa: str, target_ipa: str) -> float:
        user_ids = torch.tensor([[self.ipa_token2id[t] for t in user_ipa]], device=self.device)
        target_ids = torch.tensor([[self.ipa_token2id[t] for t in target_ipa]], device=self.device)
        with torch.no_grad():
            score = self.model(user_ids, target_ids)
        return float(score.item())
