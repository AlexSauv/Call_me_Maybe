*This project has been created as part of the 42 curriculum by alsauvan.*

**DESCRIPTION**

The purpose of Call-Me-Maybe is to understand mecanism of an Large Language Machine (LLM) and handling output produced by the Machine.

In order to do that this project is focus on helping LLM to produces a JSON file valid based on prompt and function definitions by using constrained decode method

The constrained decode method is to modify manipulate logits and keep only valid token ids, the others will get a -inf score. This methods will permit the LLM to fetch the potential next token based on the mask logit.

**Algorithm explanation**

• Algorithm explanation: Describe your constrained decoding approach in detail

The constrained decoding is based on receiving a new prompt explaining functions definitions, user prompt, the tasks the LLM has to do. This prompt will be encode and give it to the constrained decoding

The constrained decoding will be separate in 3 majors parts.

 Part 1:

The aim of this part is to get a set of int corresponding of token ids considerate as valid. It will permit to get a set of token ids that it will be looking for during constrained decoding.

part 2:

After getting the new set of valid token ids, the algorithm will apply a scoring on all token ids of the vocabulary. For all the token ids not in the set a scoring of '-inf' will be set. The others will remains unchanged.

The part will permit the LLM to be more precise during its productions.

Part 3: 

The most important part of the algorithm is this part, after applying a mask on logits. The algorithm will fetch the most potential token id, this token will be clean from parasite symbol such as 'Ġ' and add to the the produce output, the token id will be reinjected in list of valid token ids. This operation will repeat until the result be produce or the limits fixed is reached.

**INSTRUCTION**

In order to run the program properly you nedd to to have all dependencies

Please use this command

- Make install

In order to run the program use those commands:

[WITH DEFAULT INPUTS]

- Make run

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

IA used

IA tools were used for rebuild a visual representation of the README and for deeper comprehensions of topics.


• A “Resources” section listing classic references related to the topic (documentation, articles, tutorials, etc.), as well as a description of how AI was used —
specifying for which tasks and which parts of the project.
➠ Additional sections may be required depending on the project (e.g., usage
examples, feature list, technical choices, etc.).
Any required additions will be explicitly listed below.
For this project, the README.md must also include:
• Design decisions: Explain key choices in your implementation
• Performance analysis: Discuss accuracy, speed, and reliability of your solution
• Challenges faced: Document difficulties encountered and how you solved them
• Testing strategy: Describe how you validated your implementation
• Example usage: Provide clear examples of running your program