import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from typing import List
import os

class IPAPredictor(nn.Module):
    """
    Transformer-based grapheme-to-phoneme (IPA) prediction model.
    Uses a pre-trained BERT encoder and a custom decoder for IPA sequence generation.
    """
    def __init__(self, ipa_vocab_size: int, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.ipa_embedding = nn.Embedding(ipa_vocab_size, self.encoder.config.hidden_size)
        self.decoder = nn.LSTM(
            input_size=self.encoder.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, ipa_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask, decoder_input_ids, hidden=None):
        """
        Args:
            input_ids: (batch, seq_len) token ids for input text
            attention_mask: (batch, seq_len) attention mask
            decoder_input_ids: (batch, tgt_seq_len) IPA token ids for decoder input
            hidden: (h_0, c_0) for LSTM hidden state (optional)
        Returns:
            logits: (batch, tgt_seq_len, vocab_size)
        """
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        # Embed decoder input tokens
        decoder_input = self.ipa_embedding(decoder_input_ids)
        decoder_outputs, hidden = self.decoder(decoder_input, hidden)
        logits = self.fc(decoder_outputs)
        return self.softmax(logits), hidden

class IPAPredictorPipeline:
    """
    Inference pipeline for IPA prediction.
    Handles tokenization, model inference, and decoding.
    """
    def __init__(self, model_path: str, ipa_vocab: List[str]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.ipa_vocab = ipa_vocab
        self.ipa_token2id = {t: i for i, t in enumerate(ipa_vocab)}
        self.ipa_id2token = {i: t for i, t in enumerate(ipa_vocab)}
        self.model = IPAPredictor(ipa_vocab_size=len(ipa_vocab))
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, max_len: int = 32) -> str:
        """
        Predict IPA sequence for input text using greedy decoding.
        Args:
            text: input word or phrase
            max_len: maximum output length
        Returns:
            IPA string
        """
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        batch_size = 1
        # Start with <SOS> token if available, else <PAD>
        sos_token = self.ipa_token2id.get("<SOS>", 0)
        decoder_input_ids = torch.tensor([[sos_token]], device=self.device)
        hidden = None
        output_tokens = []
        for _ in range(max_len):
            logits, hidden = self.model(input_ids, attention_mask, decoder_input_ids, hidden)
            next_token_id = logits.argmax(-1)[:, -1].item()
            next_token = self.ipa_id2token[next_token_id]
            if next_token == "<EOS>":
                break
            output_tokens.append(next_token)
            decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([[next_token_id]], device=self.device)], dim=1)
        return "".join(output_tokens)
