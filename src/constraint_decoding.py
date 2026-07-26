import json
from typing import Callable


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
            self.token_to_id[token_str] = int(token_id)
            self.id_to_token[int(token_id)] = token_str

    def get_allowed_tokens(self, text: str, functions: list) -> set[int]:

        allowed_ids: set[int] = set()
        if '"name":' in text and text.count('"name":') == text.count('""'):
            for func in functions:
                func_name = func.name if hasattr(func, "name") else func["name"]
                for token, token_id in self.token_to_id.items():
                    if func_name.startswith(token.strip(' Ġ')):
                        allowed_ids.add(token_id)
        if not allowed_ids:
            return set(self.id_to_token.keys())

        return allowed_ids

    def apply_scoring(self, logits: list[float],
                      allowed_id: set[int]) -> list[float]:
        score_logits = list(logits)
        if not allowed_id:
            return score_logits
        for i in range(len(score_logits)):
            if i not in allowed_id:
                score_logits[i] = float("-inf")
        return score_logits
