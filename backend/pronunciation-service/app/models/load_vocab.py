def load_ipa_vocab(vocab_path: str):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    return vocab
