import argparse
import json
import sys
from pathlib import Path

from llm_sdk.llm_sdk import Small_LLM_Model
from src.call_me_maybe import select_func, select_params, select_prompt
from src.constraint_decoding import ConstrainedDecoder
from src.models import FuncResult, JsonFile


def set_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        type=str,
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--functions_definition",
                        type=str,
                        default="data/input/functions_definition.json")
    parser.add_argument("--output",
                        type=str,
                        default="data/output/function_calls.json")
    return parser.parse_args()


def main() -> None:
    """Main execution pipeline."""
    try:
        args = set_args()

        config = JsonFile(file_input=args.input,
                          file_func=args.functions_definition)
        all_inputs, all_func = config.load_json_files()

        model = Small_LLM_Model()
        vocab_lib = model.get_path_to_vocab_file()
        decoder = ConstrainedDecoder(vocab_lib)

        results: list[dict] = []

        for item in all_inputs:
            prompt_str = item.prompt
            prompt_text = select_prompt(all_func, prompt_str)

            encoded_tensor = model.encode(prompt_text)
            input_ids = encoded_tensor[0].tolist()

            func_def = select_func(model, decoder, input_ids, all_func)
            params = select_params(model, decoder, input_ids, func_def)

            func_res = FuncResult(
                prompt=prompt_str,
                name=func_def.name,
                parameters=params
            )
            results.append(func_res.model_dump())
            print(f"[OK] Prompt: '{prompt_str}' -> {func_def.name}({params})")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n[SUCCESS] Saved {len(results)} function calls to {output_path}")

    except Exception as e:
        print(f"[ERROR] Execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
