import json
from src.models import FuncDef


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

    def get_allowed_tokens(self, text: str,
                           functions: list[FuncDef]) -> set[int]:
        allowed_ids: set[int] = set()

        for func in functions:
            for token, token_id in self.token_to_id.items():
                clean_token = token.strip(' Ġ')
                if (func.name.startswith(clean_token)
                        or clean_token in text):
                    allowed_ids.add(token_id)

        if not allowed_ids:
            return set(self.id_to_token.keys())
        return allowed_ids

    def allowed_tokens_param(self, text: str,
                             func: FuncDef) -> set[int]:
        allowed_ids: set[int] = set()

        curr_output = text.split("Output JSON parameters:")[-1]
        if "{" not in curr_output:
            for token, token_id in self.token_to_id.items():
                if token.strip(' Ġ') == "{":
                    allowed_ids.add(token_id)
        if allowed_ids:
            return allowed_ids

        expect_keys = list(func.parameters.keys())
        for token, token_id in self.token_to_id.items():
            clean_token = token.strip(' Ġ')

            if clean_token in ["{", "}", "\"", ":", ",", " "]:
                allowed_ids.add(token_id)
                continue

            if any(p_info.type in ["number", "integer", "float"] for p_info in func.parameters.values()):
                if clean_token.isdigit() or clean_token == ".":
                    allowed_ids.add(token_id)
            if any(p_info.type == "string" for p_info in func.parameters.values()):
                if clean_token.isalpha() or clean_token == " ":
                    allowed_ids.add(token_id)

            for key in expect_keys:
                if key.startswith(clean_token) or clean_token in key:
                    allowed_ids.add(token_id)
                    break

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






# def get_allowed_tokens(self, current_text: str, schema_definition: dict) -> set[int]:
#     allowed_ids = set()
#     for token_id, token_str in self.id_to_token.items():
#         if self._is_token_valid_for_schema(current_text, token_str, schema_definition):
#             allowed_ids.add(token_id)
            
#     if not allowed_ids:
#         return set(self.id_to_token.keys())
#     return allowed_ids