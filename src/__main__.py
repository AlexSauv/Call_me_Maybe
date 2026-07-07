# from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import JsonFile
# import numpy as np
import sys


def main() -> None:
    try:
        # model = Small_LLM_Model()
        input_file = "data/input/function_calling_tests.json"
        func_file = "data/input/functions_definition.json"
        config = JsonFile(file_input=input_file, file_func=func_file)
        config.load_json_files(input_file, func_file)
        # else:
        #     raise ValueError("Not the right number of arguments")
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
