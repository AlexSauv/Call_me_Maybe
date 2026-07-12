from llm_sdk.llm_sdk import Small_LLM_Model
import json


def load_vocab_map():
    model = Small_LLM_Model()
    vocab_lib = model.get_path_to_vocab_file()
    with open(vocab_lib, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    id_to_token = {token_id: token for token, token_id in vocab.items()}
    return id_to_token

def contraint_decoding(prompt: str):
    vocab = load_vocab_map()
    model = Small_LLM_Model()
    prompt_id = model.decode(prompt)
    while True:
        logits = model.get_logits_from_input_ids(prompt_id)
        