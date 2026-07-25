from typing import Any
import numpy as np
import re
from llm_sdk.llm_sdk import Small_LLM_Model
from src.constraint_decoding import ConstrainedDecoder


def get_func(model: Small_LLM_Model,
             decoder: ConstrainedDecoder,
             prompt: str,
             functions: list[dict[str, Any]]) -> dict[str, Any]:
    if not functions:
        raise ValueError("[FUNC] No data for functions")

    high_prob_func = functions[0]
    high_score = float("-inf")
    new_prompt = f"User: {prompt}\nFunction:"
    input_ids_tensor = model.encode(new_prompt)
    input_ids = input_ids_tensor[0].tolist()
    logits = model.get_logits_from_input_ids(input_ids)

    for func in functions:
        func_name = str(func.get("name", ""))
        func_ids = model.encode(f"{func_name}")[0].tolist()
        if not func_ids:
            func_ids = model.encode(func_name)[0].tolist()
        curr_state = list(input_ids)
        score = 0.0
        for token_id in func_ids:
            logits = model.get_logits_from_input_ids(curr_state)
            cleaned_logits = decoder.apply_scoring(logits, {token_id})
            score += cleaned_logits[token_id]
        if score > high_score:
            high_score = score
            high_prob_func = func

    return high_prob_func


def get_parameters(prompt: str,
                   func_def: dict[str, Any]
                   ) -> dict[str, Any]:
    param_details = func_def.get("parameters", {})
    params: dict[str, Any] = {}
    num = re.findall(r"[-+]?\d*\.\d+|\d+", prompt)
    string = re.findall(r"'([^']*)'|\"([^\"]*)\"", prompt)
    num_index = 0
    str_index = 0
    for param_name, param_info in param_details.items():
        param_type = (param_info.get("type", "string")
                      if isinstance(param_info, dict) else "string")
        if param_type in ["number", "float", "integer"]:
            if num_index < len(num):
                value = num[num_index]
                params[param_name] = (float(value) if "."
                                      in value else int(value))
                num_index += 1
            else:
                params[param_name] = 0
        elif param_type == "string":
            if string and str_index < len(string):
                params[param_name] = string[str_index]
                str_index += 1
            else:
                params[param_name] = prompt
        elif param_type == "boolean":
            params[param_name] = "true" in prompt.lower()

    return params


def generate_call_me(model: Small_LLM_Model,
                     decoder: ConstrainedDecoder,
                     prompt: str,
                     functions: list[dict[str, Any]]
                     ) -> dict[str, Any]:
    high_prob_func = get_func(model, decoder, prompt, functions)
    func_name = str(high_prob_func.get("name", ""))
    parameters = get_parameters(prompt, high_prob_func)

    return {"prompt": prompt,
            "name": func_name,
            "parameters": parameters
            }
