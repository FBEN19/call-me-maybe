import numpy as np
from llm_sdk import Small_LLM_Model
from .constrained_decoder import generate, apply_mask
from .models import Prompt, FunctionDefinition, FunctionCall
from .tokenizer import encode


def build_prompt(prompt: Prompt, functions: list[FunctionDefinition]) -> str:
    """Build the text prompt fed to the LLM for function selection.

    The prompt lists available functions and ends with a partial token
    so the model's next token naturally selects a function name.

    Args:
        prompt: The user's natural language request.
        functions: List of available function definitions.

    Returns:
        A formatted string prompt ready for tokenization.
    """
    string = prompt.prompt + "\n\n"
    string += "Available functions:\n"
    for function in functions:
        string += f"- {function.fn_name}("
        args = []
        for arg_name in function.args_names:
            arg_type = function.args_types[arg_name]
            args.append(f"{arg_name}: {arg_type}")
        string += ", ".join(args)
        string += ")\n"
    string += "\nThe function to call is: fn"
    return string


def select_function(
    prompt: Prompt,
    functions: list[FunctionDefinition],
    model: Small_LLM_Model,
) -> FunctionCall:
    """Select the best matching function and extract its arguments.

    Uses constrained decoding to pick the function via the LLM's logits,
    then generates schema-valid JSON arguments with the same technique.

    Args:
        prompt: The user's natural language request.
        functions: List of candidate function definitions.
        model: The loaded Small_LLM_Model instance (shared across calls).

    Returns:
        A FunctionCall with the selected function name and extracted
        parameters.
    """
    built_prompt = build_prompt(prompt, functions)
    input_ids = encode(model, built_prompt)
    logits = model.get_logits_from_input_ids(input_ids)
    logits_array = np.array(logits)

    valid_tokens = []
    token_to_function = {}
    for function in functions:
        fn_tokens = encode(model, function.fn_name)
        distinctive_token = fn_tokens[1]
        if distinctive_token not in token_to_function:
            valid_tokens.append(distinctive_token)
            token_to_function[distinctive_token] = function

    masked_logits = apply_mask(logits_array, valid_tokens)
    best_token = int(np.argmax(masked_logits))
    selected = token_to_function[best_token]

    fn_name_ids = encode(model, selected.fn_name)
    input_ids_with_fn = input_ids + fn_name_ids

    parameters = generate(model, input_ids_with_fn, selected)

    return FunctionCall(
        prompt=prompt.prompt,
        name=selected.fn_name,
        parameters=parameters
    )
