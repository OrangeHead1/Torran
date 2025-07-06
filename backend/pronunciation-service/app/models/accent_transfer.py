import torch
import torch.nn as nn
from typing import List
import os

class AccentTransferModel(nn.Module):
    """
    Multi-accent transformer model for accent transfer in IPA prediction.
    """
    def __init__(self, ipa_vocab_size: int, accent_count: int, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.accent_embeddings = nn.Embedding(accent_count, hidden_size)
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, ipa_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_embeds, accent_id, decoder_input, hidden=None):
        accent_embed = self.accent_embeddings(accent_id).unsqueeze(1)
        encoder_input = input_embeds + accent_embed
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, hidden)
        decoder_outputs, _ = self.decoder(decoder_input, encoder_hidden)
        logits = self.fc(decoder_outputs)
        return self.softmax(logits)

class AccentTransferPipeline:
    def __init__(self, model_path: str, ipa_vocab: List[str], accent2id: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ipa_vocab = ipa_vocab
        self.ipa_token2id = {t: i for i, t in enumerate(ipa_vocab)}
        self.ipa_id2token = {i: t for i, t in enumerate(ipa_vocab)}
        self.accent2id = accent2id
        self.model = AccentTransferModel(ipa_vocab_size=len(ipa_vocab), accent_count=len(accent2id))
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_embeds, accent: str, max_len: int = 32):
        accent_id = torch.tensor([self.accent2id[accent]], device=self.device)
        batch_size = 1
        hidden_size = input_embeds.size(-1)
        decoder_input = torch.zeros((batch_size, 1, hidden_size), device=self.device)
        hidden = None
        output_tokens = []
        for _ in range(max_len):
            logits = self.model(input_embeds, accent_id, decoder_input, hidden)
            next_token_id = logits.argmax(-1)[:, -1].item()
            next_token = self.ipa_id2token[next_token_id]
            if next_token == "<EOS>":
                break
            output_tokens.append(next_token)
            next_input = torch.zeros((batch_size, 1, hidden_size), device=self.device)
            decoder_input = torch.cat([decoder_input, next_input], dim=1)
        return "".join(output_tokens)
