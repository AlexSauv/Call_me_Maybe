import sys
try:
    import argparse
    import json
    from pathlib import Path
    from llm_sdk import Small_LLM_Model  # type: ignore
    from src.call_me_maybe import call_me_maybe
    from src.constraint_decoding import ConstrainedDecoder
    from src.models import JsonFile
except KeyboardInterrupt as e:
    print(f"The program has been closed: {e}")
    sys.exit(1)
except ImportError as e:
    print(f"[ERROR][IMPORT] {e}")
    sys.exit(1)


def set_args() -> argparse.Namespace:
    """
        parse arguments given for Json files
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        type=str,
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--functions_definition",
                        type=str,
                        default="data/input/functions_definition.json")
    parser.add_argument("--output",
                        type=str,
                        default="data/output/function_calling_results.json")
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

        results = call_me_maybe(model, decoder, all_inputs, all_func)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"[ERROR] Execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
