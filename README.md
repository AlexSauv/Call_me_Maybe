*This project has been created as part of the 42 curriculum by alsauvan.*

**DESCRIPTION**

Call-Me-Maybe is a Python-based project designed to bridge the gap between natural language prompts and machine-executable code through Function Calling. Using a small Large Language Model (Qwen/Qwen3-0.6B), the project guarantees 100% structurally and semantically valid JSON outputs without relying on prompt-hoping or heuristics.

By implementing a custom Constrained Decoding mechanism, the engine manipulates model logits token-by-token, forcing the generation to strictly adhere to predefined function schemas and expected argument types.

**Algorithm explanation**

The decoding pipeline transforms natural language requests into structured JSON calls through three core steps implemented in ConstrainedDecoder and call_me_maybe.py:

Token Filtering (get_allowed_tokens):
The decoder maps the vocabulary file (vocab.json) to identify which token IDs represent valid continuations for the current prefix (whether matching a function name, string value, boolean, or numerical bounds).

Logit Masking (apply_scoring):
For any given generation step, the raw logits returned by the model are modified using NumPy. All token IDs outside the allowed whitelist are assigned a score of -inf, leaving only valid tokens active.

Token Selection & Iteration:
Using np.argmax, the algorithm picks the highest-scoring valid token, cleans BPE artifacts (such as Ġ and Ċ), appends it to the running input sequence, and repeats the process until the parameter or function structure is fully complete.

**Design decisions**

Split Parameter Parsing: The project splits generation logic into dedicated handlers for each type (select_func, select_string_params, select_number_params, and select_boolean_param). This modularity reduces hallucination and constrains the token search space exclusively to what the schema permits.

Pydantic Validation: All input files (functions_definition.json and function_calling_tests.json) and data payloads are strictly validated at runtime using Pydantic models (FuncDef, PromptInput, JsonFile), ensuring early failure and descriptive error handling.

Dependency Management: Built with uv for fast, reliable environment and package management, cleanly integrated via a comprehensive Makefile.

**Performance analysis**

The algorithm relie on accuracy but the speed will be impact by it because it costs more on time to find the most potential ouput result 

**Struggling part**

The most difficult part encountered was to help the LLM to choose the right parameters specifity in the user prompt, however to solve it the algoritm has been split for each specific part. This method permits to the LLM to reduce the list of token ids allowed and to focus only on this particulary part.

**Testing Strategy**
To validate the implementation:

Schema Validation: Pydantic models verify that loaded definitions and final JSON structures map correctly.

Edge-Case Handling: Tested with negative numbers, strings, and missing or malformed inputs to ensure robust error handling without crashing.

**INSTRUCTION**

In order to run the program properly you need to have all dependencies

Please use this command

- Make install
- uv sync

In order to run the program use those commands:

[WITH DEFAULT INPUTS]

- Make run
- uv run python -m src

[WITH SPECIFIC ARGUMENTS]

You can change for any file paths this following example

uv run python -m src
--functions_definition data/input/functions_definition.json
--input data/input/function_calling_tests.json
--output data/output/function_calls.json

**RESOURCES**

- https://docs.lm-kit.com/lm-kit-net/guides/glossary/logits.html

- https://huggingface.co/Qwen/Qwen3-0.6B

- https://medium.com/@rosgluk/constraining-llms-with-structured-output-ollama-qwen3-python-or-go-2f56ff41d720

- https://www.geeksforgeeks.org/python/json-load-in-python/

- https://medium.com/@adkananthi/logits-as-confidence-the-hidden-power-ai-engineers-need-to-unlock-in-llms-and-vlms-194d512c31f2

IA used:

IA tools were used for rebuild a visual representation of the README and for deeper comprehensions of topics.

• Testing strategy: Describe how you validated your implementation
• Example usage: Provide clear examples of running your program