from typing import Any
from llm_sdk.llm_sdk import Small_LLM_Model
from src.constraint_decoding import ConstrainedDecoder
from src.models import FuncDef, FuncResult, PromptInput


def get_func(model: Small_LLM_Model,
             decoder: ConstrainedDecoder,
             prompt: str,
             functions: list[FuncDef]) -> FuncDef:
    if not functions:
        raise ValueError("[FUNC] No data for functions")
    new_prompt = f"Instruction: {prompt}\nTarget Function Name:"
    input_ids = model.encode(new_prompt)[0].tolist()
    found = False
    result = functions[0]
    for _ in range(100):

        logits = model.get_logits_from_input_ids(input_ids)

        allowed_ids = decoder.get_allowed_tokens(input_ids, functions)

        cleaned_logits = decoder.apply_scoring(logits, allowed_ids)

        next_token_id = int(max(range(len(cleaned_logits)),
                                key=lambda i: cleaned_logits[i]))
        input_ids.append(next_token_id)

        decode_token = model.decode([next_token_id])
        if '\n' in decode_token:
            break
    final_output_text = model.decode(input_ids)
    clean_name = final_output_text.split("Target Function Name:")[-1].strip().strip('"\'')
    for func in functions:
        if clean_name == func.name:
            return func
    return functions[0]


def get_parameters(prompt: str,
                   func_def: FuncDef
                   ) -> dict[str, Any]:
    param_details = func_def.parameters
    params: dict[str, Any] = {}

    words = prompt.split()
    for param_name, param_info in param_details.items():
        param_type = getattr(param_info, "type", "string")
        if isinstance(param_info, dict):
            param_type = param_info.get("type", "string")

        if param_type in ["number", "float", "integer"]:
            num = []
            for word in words:
                new_word = "".join(c for c in word if c.isdigit() or c == ".")
                if new_word:
                    try:
                        value = float(word) if "." in word else int(word)
                        num.append(value)
                        break
                    except ValueError:
                        continue
            if num:
                if len(num) == 1:
                    params[param_name] = num[0]
                else:
                    index = list(param_details.keys()).index(param_name)
                    params[param_name] = num[index] if index < len(num) else num[0]
            else:
                params[param_name] = 0
        elif param_type == "string":
            params[param_name] = prompt
        elif param_type == "boolean":
            params[param_name] = "true" in prompt.lower()

    return params


def generate_call_me(model: Small_LLM_Model,
                     decoder: ConstrainedDecoder,
                     user_text: PromptInput,
                     functions: list[dict[str, Any]]
                     ) -> FuncResult:
    prompt = (user_text.prompt if hasattr(user_text,
                                          "prompt") else str(user_text))

    high_prob_func = get_func(model, decoder, prompt, functions)
    func_name = high_prob_func.name
    parameters = get_parameters(user_text.prompt, high_prob_func)

    result = FuncResult(prompt=prompt, name=func_name, parameters=parameters)
    return result
