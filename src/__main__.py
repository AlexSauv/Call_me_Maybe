from llm_sdk.llm_sdk import Small_LLM_Model
from src.constraint_decoding import ConstrainedDecoder
from src.models import JsonFile
import argparse
import json
from pathlib import Path
from src.call_me_maybe import generate_call_me


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
    try:
        args = set_args()
        args_input = args.input
        args_func = args.functions_definition

        config = JsonFile(file_input=args_input,
                          file_func=args_func)
        all_inputs, all_func = config.load_json_files()

        model = Small_LLM_Model()
        vocab_lib = model.get_path_to_vocab_file()
        decoder = ConstrainedDecoder(vocab_lib)

        result = []
        for item in all_inputs:
            res = generate_call_me(model, decoder, item, all_func)
            result.append(res)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            json.dump([res.model_dump() for res in result], f, indent=2)

    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
