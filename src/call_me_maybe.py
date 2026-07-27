from typing import Any
import json
import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from src.constraint_decoding import ConstrainedDecoder
from src.models import FuncDef, FuncResult, PromptInput
from src.utils import find_all_strings, find_all_numbers


def select_prompt_func(functions: list[FuncDef], prompt: str) -> str:
    func_details = "\n".join(f" - {f.name}: {f.description}"
                             for f in functions)
    return (
        f"You must pick the correct function name for this request.\n"
        f"Available functions:\n{func_details}\n\n"
        f'Request: "{prompt}"\n\n'
        f"The correct function name is: ")


def select_prompt_params(prompt: str,
                         func_name: str) -> str:
    return (
        "You must pick the correct parameters"
        f" for function '{func_name}' based on the request.\n"
        f"Request: \"{prompt}\"\n"
        f"Output JSON parameters:"
    )


def select_func(model: Small_LLM_Model,
                decoder: ConstrainedDecoder,
                prompt: str,
                functions: list[FuncDef]) -> FuncDef:
    if not functions:
        raise ValueError("[FUNC] No data for functions")
    new_prompt = select_prompt_func(functions, prompt)
    encoded_tensor = model.encode(new_prompt)
    input_ids = encoded_tensor[0].tolist()

    for _ in range(30):

        logits = model.get_logits_from_input_ids(input_ids)

        decode_text = model.decode(input_ids)

        allowed_token_ids = decoder.get_allowed_tokens(decode_text, functions)

        cleaned_logits = decoder.apply_scoring(logits, allowed_token_ids)

        next_token_id = int(max(range(len(cleaned_logits)),
                                key=lambda i: cleaned_logits[i]))
        input_ids.append(next_token_id)

        curr_output = model.decode(input_ids)
        generate = curr_output.split("The correct function name is:")[-1].strip().strip('"\'')
        if any(func.name == generate for func in functions):
            break
        decode_token = model.decode([next_token_id])
        if "\n" in decode_token and len(generate) > 0:
            break
    final_output_text = model.decode(input_ids)
    clean_name = final_output_text.split("The correct function name is:")[-1].strip().strip('"\'')
    clean_name = clean_name.replace(" ", "_")
    for func in functions:
        if clean_name == func.name:
            return func
    for func in functions:
        if func.name in clean_name or clean_name in func.name:
            return func
    return functions[0]


def select_param(model: Small_LLM_Model,
                 decoder: ConstrainedDecoder,
                 user_prompt: str,
                 func_def: FuncDef,
                 ) -> dict[str, Any]:
    new_prompt = select_prompt_params(user_prompt, func_def.name)

    encoded_tensor = model.encode(new_prompt)
    input_ids = encoded_tensor[0].tolist()

    for _ in range(30):
        logits = model.get_logits_from_input_ids(input_ids)
        decode_text = model.decode(input_ids)

        allowed_ids = decoder.allowed_tokens_param(decode_text, func_def)
        cleaned_logits = decoder.apply_scoring(logits, allowed_ids)

        next_token_id = int(np.argmax(cleaned_logits))
        input_ids.append(next_token_id)

        curr_output = model.decode(input_ids)
        print(curr_output)
        # if "}" in curr_output.split("Output JSON parameters:")[-1]:
        #     break
    final_output_text = model.decode(input_ids)
    raw_json_str = final_output_text.split("Output JSON parameters:")[-1].strip()
    try:
        start_idx = raw_json_str.find('{')
        end_idx = raw_json_str.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = raw_json_str[start_idx:end_idx+1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return {}

    # param_details = func_def.parameters
    # params: dict[str, Any] = {}
    # num_found = 0
    # str_found = 0

    # for param_name, param_info in param_details.items():
    #     param_type = getattr(param_info, "type", "string")
    #     if isinstance(param_info, dict):
    #         param_type = param_info.get("type", "string")

    #     if param_type in ["number", "float", "integer"]:
    #         num = find_all_numbers(user_prompt)
    #         if num:
    #             if num_found < len(num):
    #                 params[param_name] = num[num_found]
    #                 num_found += 1
    #         else:
    #             raise ValueError("[PARAM] parameter data not found")
    #     elif param_type in ["string", "regex"]:
    #         strings = find_all_strings(user_prompt)
    #         if str_found < len(strings):
    #             params[param_name] = strings[str_found]
    #             str_found += 1
    #         else:
    #             params[param_name] = user_prompt
    #     elif param_type == "boolean":
    #         params[param_name] = "true" in user_prompt.lower()

    # return params


def generate_call_me(model: Small_LLM_Model,
                     decoder: ConstrainedDecoder,
                     user_text: PromptInput,
                     functions: list[dict[str, Any]]
                     ) -> FuncResult:
    prompt = (user_text.prompt if hasattr(user_text,
                                          "prompt") else str(user_text))

    high_prob_func = select_func(model, decoder, prompt, functions)
    func_name = high_prob_func.name

    params = select_param(model, decoder, prompt, high_prob_func)

    return FuncResult(prompt=prompt, name=func_name, parameters=params)
