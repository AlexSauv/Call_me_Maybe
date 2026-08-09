import json
from src.models import FuncDef
import numpy as np


class ConstrainedDecoder:
    def __init__(self, vocab_file: str) -> None:
        self.vocab_file = vocab_file
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._load_vocab()

    def _load_vocab(self):
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        for token_str, token_id in vocab.items():
            tok_id = int(token_id)
            self.token_to_id[token_str] = tok_id
            self.id_to_token[tok_id] = token_str

    def get_allowed_tokens(self, generate_data: str, valid_tokens: list[str]) -> set[int]:
        allowed_ids: set[int] = set()
        for token, token_id in self.token_to_id.items():
            next_prob = generate_data + token
            for valid_token in valid_tokens:
                if valid_token.startswith(next_prob):
                    allowed_ids.add(token_id)
                    break
        if not allowed_ids:
            raise ValueError("[CONSTRAINT] No token ids allowed.")
        return allowed_ids

    def apply_scoring(self, logits: list[float],
                      allowed_ids: set[int]) -> list[float]:
        if not allowed_ids:
            return logits
        masked_logits = np.full(len(logits), -float('inf'))
        allowed = list(allowed_ids)
        for token_id in allowed:
            if 0 <= token_id < len(logits):
                masked_logits[token_id] = logits[token_id]
        return masked_logits.tolist()
