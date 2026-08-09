import re
from typing import Any
import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from src.constraint_decoding import ConstrainedDecoder
from src.models import FuncDef

def append_text_tokens(model: Small_LLM_Model, input_ids: list[int], text: str) -> None:
    """Encodes a fixed text string and appends its token IDs directly to input_ids."""
    encoded_tensor = model.encode(text)
    ids = encoded_tensor[0].tolist()
    input_ids.extend(ids)

def clean_token_text(token_str: str) -> str:
    """Normalizes BPE special space/newline representation to plain text."""
    return token_str.replace("Ġ", " ").replace("Ċ", "\n")

def select_prompt(functions: list[FuncDef], user_prompt: str) -> str:
    """Builds a strictly generic prompt derived only from function signatures."""
    tools_list = []
    for f in functions:
        args = ", ".join(f"{name}: {info.type}" for name, info in f.parameters.items())
        tools_list.append(f"- {f.name}({args}): {f.description}")
    
    tools_formatted = "\n".join(tools_list)

    return (
        f"You are a precise function-calling assistant. "
        f"Extract literal values and positive patterns directly from the user prompt.\n\n"
        f"Available Functions:\n"
        f"{tools_formatted}\n\n"
        f"User Input: {user_prompt}\n\n"
        f'JSON Output:\n{{"prompt": "{user_prompt}", "name": "'
    )


def select_func(model: Small_LLM_Model,
                decoder: ConstrainedDecoder,
                input_ids: list[int],
                functions: list[FuncDef]) -> FuncDef:
    if not functions:
        raise ValueError("[FUNC] No data for functions")

    func_names = [func.name for func in functions]
    high_prob_func = ""
    while True:
        matching_funcs = [f for f in functions if f.name == high_prob_func]
        prefix_matches = [f for f in functions if f.name.startswith(high_prob_func)]

        if matching_funcs and len(prefix_matches) == 1:
            return matching_funcs[0]

        logits = model.get_logits_from_input_ids(input_ids)
        allowed_token_ids = decoder.get_allowed_tokens(high_prob_func, func_names)
        masked_logits = decoder.apply_scoring(logits, allowed_token_ids)
        next_token_id = int(np.argmax(masked_logits))

        token = decoder.id_to_token[next_token_id]
        clean_token = clean_token_text(token)
        high_prob_func += clean_token
        input_ids.append(next_token_id)


def select_boolean_param(model: Small_LLM_Model,
                         decoder: ConstrainedDecoder,
                         input_ids: list[int]) -> bool:
    """Constrains generate 'true' or 'false' using LLM logits."""
    logits = model.get_logits_from_input_ids(input_ids)
    bool_targets = ["true", "false"]

    allowed_ids = decoder.get_allowed_tokens("", bool_targets)
    masked_logits = decoder.apply_scoring(logits, allowed_ids)
    next_token_id = int(np.argmax(masked_logits))

    input_ids.append(next_token_id)

    return True if decoder.id_to_token[next_token_id] == "true" else False
    

def select_string_params(
        model: Small_LLM_Model,
        decoder: ConstrainedDecoder,
        input_ids: list[int]
        ) -> str:
    append_text_tokens(model, input_ids, '"')
    high_prob_str = ""

    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        allowed_ids = {
            tid for tok, tid in decoder.token_to_id.items()
            if "\n" not in tok and "Ċ" not in tok
            and not (high_prob_str == "" and clean_token_text(tok).startswith('"'))
        }
        masked_logits = decoder.apply_scoring(logits, allowed_ids)
        next_token_id = int(np.argmax(masked_logits))

        token = decoder.id_to_token[next_token_id]
        clean_token = clean_token_text(token)

        if '"' in clean_token:
            part = clean_token.split('"')[0]
            high_prob_str += part
            input_ids.append(next_token_id)
            break
        high_prob_str += clean_token
        input_ids.append(next_token_id)

        if len(high_prob_str) > 200:
            append_text_tokens(model, input_ids, '"')
            break

    return high_prob_str


def select_number_params(model: Small_LLM_Model,
                         decoder: ConstrainedDecoder,
                         input_ids: list[int],
                         is_float: bool = True
                         ) -> float | int:
    high_num_prob = ""
    valid_chars = set("0123456789.-" if is_float else set("0123456789-"))

    while True:
        logits = model.get_logits_from_input_ids(input_ids)

        allowed_ids = set()
        for tok, tid in decoder.token_to_id.items():
            clean_tok = clean_token_text(tok).strip()
            if clean_tok and all(c in valid_chars for c in clean_tok):
                prob = high_num_prob + clean_tok
                if prob in ("-", ".") or prob.replace(".", "", 1).replace("-", "", 1).isdigit():
                    allowed_ids.add(tid)

        if not allowed_ids:
            break

        masked_logits = decoder.apply_scoring(logits, allowed_ids)
        next_token_id = int(np.argmax(masked_logits))

        token = decoder.id_to_token[next_token_id]
        clean_token = clean_token_text(token).strip()

        high_num_prob += clean_token
        input_ids.append(next_token_id)

        next_logits = model.get_logits_from_input_ids(input_ids)
        high_prob_token_ids = int(np.argmax(next_logits))
        high_prob_token = clean_token_text(decoder.id_to_token[high_prob_token_ids])
            
        if high_num_prob not in ("", "-", ".") and any(c in high_prob_token for c in [",", "}", " ", '"', "\n"]):
            break
    try:
        return float(high_num_prob) if is_float else int(high_num_prob)
    except ValueError:
        raise ValueError(f"[NUM] Invalid number generated: {high_num_prob}")


def select_params(
    model: Small_LLM_Model,
    decoder: ConstrainedDecoder,
    input_ids: list[int],
    func: FuncDef
) -> dict[str, Any]:
    """Generates schema-compliant parameters entirely through LLM constrained decoding."""
    params_result: dict[str, Any] = {}
    param_items = list(func.parameters.items())

    append_text_tokens(model, input_ids, '", "parameters": {')

    for i, (param_name, param_info) in enumerate(param_items):
        append_text_tokens(model, input_ids, f'"{param_name}": ')
        p_type = param_info.type

        if p_type == "string":
            val = select_string_params(model, decoder, input_ids)
        elif p_type == "number":
            val = select_number_params(model, decoder, input_ids, is_float=True)
        elif p_type == "float":
            val = select_number_params(model, decoder, input_ids, is_float=True)
        elif p_type == "integer":
            val = select_number_params(model, decoder, input_ids, is_float=False)
        elif p_type == "boolean":
            val = select_boolean_param(model, decoder, input_ids)
        elif p_type == "null":
            append_text_tokens(model, input_ids, "null")
            val = None
        else:
            val = select_string_params(model, decoder, input_ids)

        params_result[param_name] = val

        if i < len(param_items) - 1:
            append_text_tokens(model, input_ids, ", ")

    append_text_tokens(model, input_ids, "}}")
    return params_result



# def generate_call_me(model: Small_LLM_Model,
#                      decoder: ConstrainedDecoder,
#                      user_text: PromptInput,
#                      functions: list[dict[str, Any]]
#                      ) -> FuncResult:
#     prompt = (user_text.prompt if hasattr(user_text,
#                                           "prompt") else str(user_text))
#     func = select_func(model, decoder,, functions)
#     func_name = func.name

#     # params = select_params(model, decoder, prompt, func)

#     return FuncResult(prompt=prompt, name=func_name, parameters=params)
