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


    def get_allowed_tokens(self, user_prompt, functions: list[FuncDef]) -> set[int]:
        allowed_ids: set[int] = set()

        for func in functions:
            for token, token_id in self.token_to_id.items():
                clean_token = token.strip(' Ġ')
                if (func.name.startswith(clean_token)
                        or clean_token in user_prompt):
                    allowed_ids.add(token_id)

        if not allowed_ids:
            raise ValueError("[CONSTRAINT] No token ids allowed.")
        return allowed_ids

    def allowed_tokens_param(self, curr_output: str,
                             func: FuncDef) -> set[int]:
        allowed_ids: set[int] = set()

    def _check_param_key(self,
                         curr_output: str,
                         curr_token: str,
                         func: FuncDef):
        clean_token = curr_token.strip(' Ġ')
        stripped_output = curr_output.strip()
        if stripped_output.endswith("{"):
            return curr_token == '"' or curr_token in [' ', ' Ġ']

        if stripped_output.count('"') % 2 != 0 and not stripped_output.endswith(":"):
            valid_keys = func.parameters.keys()
            for key in valid_keys:
                if (clean_token in key
                        or key.startswith(clean_token)
                        or clean_token == '"'):
                    return True
        return False

    def apply_scoring(self, logits: list[float],
                      allowed_id: set[int]) -> list[float]:
        score_logits = list(logits)
        if not allowed_id:
            return score_logits
        for i in range(len(score_logits)):
            if i not in allowed_id:
                score_logits[i] = float("-inf")
        return score_logits
