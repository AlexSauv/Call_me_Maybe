from llm_sdk.llm_sdk import Small_LLM_Model


def load_vocab_map(sdk_model: Small_LLM_Model):
    vocab = sdk_model.get_path_to_vocab_file()
    