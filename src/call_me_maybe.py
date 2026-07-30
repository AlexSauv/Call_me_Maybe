from typing import Any
import json
import re
import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from src.constraint_decoding import ConstrainedDecoder
from src.models import FuncDef, FuncResult, PromptInput


def select_prompt(functions: list[FuncDef], user_prompt: str) -> str:
    func_details = "\n".join(f" - {f.name}: {f.description}"
                             for f in functions)
    return (
        f"Your task is to map the user's natural language input to the "
        f"exact function that can answer it.\n\n"
        f"AVAILABLE FUNCTIONS:\n{func_details}\n\n"
        f"RULES:\n"
        f"1. Select the most appropriate function from the list above.\n"
        f"2. Extract the required parameters from the USER INPUT.\n"
        f"3. Output strictly valid JSON.\n"
        f"4. If a string is given, extract it from the quotes.\n\n"
        f'USER INPUT:\n"{user_prompt}"\n\n'
        f'{{"prompt":"{user_prompt}","name":"'
    )


def select_func(model: Small_LLM_Model,
                decoder: ConstrainedDecoder,
                input_ids: list[int],
                functions: list[FuncDef]) -> FuncDef:
    if not functions:
        raise ValueError("[FUNC] No data for functions")

    func_found = False
    func_names = [func.name for func in functions]
    logits = model.get_logits_from_input_ids(input_ids)
    high_prob_func = ""
    while high_prob_func not in func_names:
        allowed_token_ids = decoder.get_allowed_tokens(high_prob_func, func_names)
        if not allowed_token_ids:
            break

        masked_logits = decoder.apply_scoring(logits, allowed_token_ids)
        next_token_id = int(np.argmax(masked_logits))

        token = decoder.id_to_token[next_token_id]
        high_prob_func += token
        input_ids.append(next_token_id)
        logits = model.get_logits_from_input_ids(input_ids)

        for func in functions:
            if high_prob_func == func.name:
                result = func
                func_found = True
                break
    if not func_found:
        raise ValueError("[FUNCTION] No corresponding Function name found.")
    return result


def allowed_token_params(model: Small_LLM_Model,
                         decoder: ConstrainedDecoder,
                         target_text: str,
                         input_ids: list[int]):
    remain_text = target_text
    while remain_text:
        logits = model.get_logits_from_input_ids(input_ids)
        allowed_ids = []
        for token, token_id in decoder.token_to_id.items():
            if remain_text.startswith(token):
                allowed_ids.append(token_id)
        if not allowed_ids:
            break

        masked_logits = decoder.apply_scoring(logits, set(allowed_ids))
        next_token_id = int(np.argmax(masked_logits))

        token = decoder.id_to_token[next_token_id]
        input_ids.append(next_token_id)
        if remain_text.startswith(token):
            remain_text = remain_text[len(token):]
        else:
            break

def select_params(model: Small_LLM_Model,
                  decoder: ConstrainedDecoder,
                  input_ids: list[int],
                  func: FuncDef,
                  ) -> dict[str, Any]:

    high_prob_params: dict[str, Any] = {}
    params = list(func.parameters.items())

    allowed_token_params(model, decoder, '", "parameters": {', input_ids)
    for i, (param_name, param_info) in enumerate(params):
        allowed_token_params(model, decoder, f'"{param_name}"', input_ids)
        param_type = param_info.type
        if param_type == "string":
            print("rework to do")
            value = "NOT DONE"
        elif param_type == "number":
            num = select_numbers(model, decoder, input_ids)
            value = float(num)
        elif param_type == "integer":
            num = select_numbers(model, decoder, input_ids)
            value = int(num)
            value = "NOT DONE"
        elif param_type == "boolean":
            print("rework to do")
            value = "NOT DONE"
        elif param_type == "null":
            print("rework to do")
            value = "NOT DONE"
        high_prob_params[param_name] = value
        if i < len(params) - 1:
            allowed_token_params(model, decoder, ",", input_ids)

    allowed_token_params(model, decoder, "}}", input_ids)
    return high_prob_params


def select_numbers(model: Small_LLM_Model,
                   decoder: ConstrainedDecoder,
                   input_ids: list[int]):
    high_prob_num = ""
    num_pattern = re.compile(r'^-?\d+(\.\d+)?$')
    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        valid_ids = []
        for token, token_id in decoder.token_to_id:
            candidate = high_prob_num + token
            valid = re.match(r'^-?[\d.]*$', candidate)
            if valid and candidate not in ("-", "."):
                valid_ids.append(token_id)
        if not valid_ids:
            break
        masked_logits = decoder.apply_scoring(logits, valid_ids)
        next_token_id = int(np.argmax(masked_logits))
        token = decoder.token_to_id[next_token_id]

        candidate = high_prob_num + token
        if not re.match(r'^-?[\d.]*$', candidate):
            break

        high_prob_num += token
        input_ids.append(next_token_id)

        if num_pattern.match(high_prob_num):
            if '.' in high_prob_num:
                break
            next_logits = model.get_logits_from_input_ids(input_ids)
            stop_ids = [
                token_id for token, token_id in decoder.token_to_id
                if token in (',', '}', ' ', '\n')
            ]
            best_stop = max(stop_ids, key=lambda i: next_logits[i])
            best_any = int(np.argmax(next_logits))
            stop_close = next_logits[best_stop] > next_logits[best_any] - 1.0
            if best_any in stop_ids or stop_close:
                break

    return high_prob_num


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
