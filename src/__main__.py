from llm_sdk.llm_sdk import Small_LLM_Model
from src.constraint_decoding import ConstrainedDecoder
from src.models import JsonFile
import argparse


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
        model = Small_LLM_Model()
        vocab_lib = model.get_path_to_vocab_file()
        decoder = ConstrainedDecoder(vocab_lib)
        args = set_args()
        args_input = args.input
        args_func = args.functions_definition
        config = JsonFile(file_input=args_input, file_func=args_func)
        all_inputs, all_func = config.load_json_files()
        print(all_inputs)

    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
