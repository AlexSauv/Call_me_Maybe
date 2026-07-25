import json


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

    # def get_token_ids(self, text: str) -> list[int]:
    #     token_ids = []
    #     for token_str, token_id in self.token_to_id.items():
    #         clean_token = token_str.lstrip("Ġ").lstrip(" ")
    #         if clean_token and (text.startswith(clean_token)
    #                             or clean_token.startswith(text)):
    #             token_ids.append(token_id)
    #     return token_ids

    def apply_scoring(self, logits: list[float], allowed_id: set[int]) -> list[float]:
        score_logits = list(logits)
        if not allowed_id:
            return score_logits
        for i in range(len(score_logits)):
            if i not in allowed_id:
                score_logits[i] = float("-inf")
        return score_logits
