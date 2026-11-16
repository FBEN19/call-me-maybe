import json
from typing import Any
import numpy as np
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition
from .tokenizer import encode

MAX_TOKENS = 500
MAX_VALUE_CONTENT_TOKENS = 50


def apply_mask(logits: np.ndarray,
               valid_tokens: list[int]) -> np.ndarray:
    """Apply a mask to logits, setting invalid token scores to -inf.

    Args:
        logits: Raw logit array from the model.
        valid_tokens: List of token IDs that are allowed at this step.

    Returns:
        Masked logit array where invalid tokens have score -inf.
    """
    mask = np.full_like(logits, -np.inf)
    mask[valid_tokens] = 0
    return logits + mask


def get_valid_tokens(state: str, current_arg: str, current_arg_type: str,
                     current_arg_tokens: list[int],
                     vocabulary: dict[str, int],
                     args_remaining: int,
                     value_content_count: int = 0) -> list[int]:
    """Return the list of valid token IDs for the current decoding state.

    Args:
        state: Current state of the JSON generation state machine.
        current_arg: Name of the argument currently being generated.
        current_arg_type: Type string of the current argument.
        current_arg_tokens: Remaining token IDs for the current key name.
        vocabulary: Mapping from token string to token ID.
        args_remaining: Number of arguments still to be generated.
        value_content_count: Number of content tokens emitted so far
            for the current string value.

    Returns:
        List of token IDs valid at this decoding step.

    Raises:
        ValueError: If no valid tokens exist for the given state.
    """
    valid = []
    for token_str, token_id in vocabulary.items():
        if state == "START":
            if token_str == "{":
                valid.append(token_id)
        elif state == "EXPECT_KEY_OPEN_QUOTE":
            if token_str == '"':
                valid.append(token_id)
        elif state == "EXPECT_KEY_NAME":
            valid.append(current_arg_tokens[0])
            break
        elif state == "EXPECT_KEY_CLOSE_QUOTE":
            if token_str == '"':
                valid.append(token_id)
        elif state == "EXPECT_COLON":
            if token_str == ":":
                valid.append(token_id)
        elif state == "EXPECT_VALUE":
            if current_arg_type in ("float", "int"):
                if token_str.lstrip("-").replace(".", "", 1).isdigit():
                    valid.append(token_id)
            elif current_arg_type == "str":
                if token_str == '"':
                    valid.append(token_id)
            elif current_arg_type == "bool":
                if token_str in ("true", "false"):
                    valid.append(token_id)
        elif state == "EXPECT_VALUE_CONTINUE":
            if current_arg_type in ("float", "int"):
                if token_str.lstrip("-").replace(".", "", 1).isdigit():
                    valid.append(token_id)
                elif args_remaining > 0 and token_str == ",":
                    valid.append(token_id)
                elif args_remaining == 0 and token_str == "}":
                    valid.append(token_id)
        elif state == "EXPECT_VALUE_CONTENT":
            if value_content_count >= MAX_VALUE_CONTENT_TOKENS:
                if token_str == '"':
                    valid.append(token_id)
            else:
                has_control = 'Ċ' in token_str or 'ĉ' in token_str
                if '"' not in token_str and not has_control:
                    valid.append(token_id)
                elif token_str == '"':
                    valid.append(token_id)
        elif state == "EXPECT_VALUE_CLOSE_QUOTE":
            if token_str == '"':
                valid.append(token_id)
        elif state == "EXPECT_COMMA":
            if args_remaining > 0:
                if token_str == ",":
                    valid.append(token_id)
            else:
                if token_str == "}":
                    valid.append(token_id)
        elif state == "END":
            if token_str == "}":
                valid.append(token_id)
    if not valid:
        raise ValueError(f"No valid tokens found for state {state}")
    return valid


def clean_token(token_str: str) -> str:
    """Convert tokenizer glyphs to their actual string representations.

    Args:
        token_str: Raw token string from the vocabulary.

    Returns:
        The token string with Unicode glyphs replaced by their characters.
    """
    return token_str.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")


def generate(model: Small_LLM_Model, input_ids: list[int],
             function: FunctionDefinition) -> dict[str, Any]:
    """Generate the JSON parameters object for a function call.

    Uses a state-machine-based constrained decoding loop to guarantee
    that the output is valid JSON matching the function's argument schema.

    Args:
        model: The loaded Small_LLM_Model instance.
        input_ids: Tokenized prompt IDs including the selected function name.
        function: The function definition describing expected arguments.

    Returns:
        A dictionary mapping argument names to their extracted values.
    """
    vocab_path = model.get_path_to_vocabulary_json()
    with open(vocab_path, "r") as f:
        vocabulary = json.load(f)

    vocab_by_id = {v: k for k, v in vocabulary.items()}

    state = "START"
    args_to_process = list(function.args_names)
    current_arg: str = ""
    current_arg_type: str = ""
    current_arg_tokens: list[int] = []
    generated_token_ids = []
    value_content_count = 0
    token_count = 0

    while state != "END":
        if token_count > MAX_TOKENS:
            print(f"Warning: max tokens reached at state={state}, stopping.")
            break
        token_count += 1

        logits = model.get_logits_from_input_ids(input_ids)
        logits_array = np.array(logits)
        valid_tokens = get_valid_tokens(state, current_arg, current_arg_type,
                                        current_arg_tokens, vocabulary,
                                        len(args_to_process),
                                        value_content_count)
        masked_logits = apply_mask(logits_array, valid_tokens)
        best_token_id = int(np.argmax(masked_logits))
        best_token_str = vocab_by_id.get(best_token_id, "")
        generated_token_ids.append(best_token_id)
        input_ids.append(best_token_id)

        if state == "START":
            state = "EXPECT_KEY_OPEN_QUOTE"
            current_arg = args_to_process.pop(0)
            current_arg_type = function.args_types[current_arg]
            current_arg_tokens = encode(model, current_arg)
        elif state == "EXPECT_KEY_OPEN_QUOTE":
            state = "EXPECT_KEY_NAME"
        elif state == "EXPECT_KEY_NAME":
            current_arg_tokens.pop(0)
            if not current_arg_tokens:
                state = "EXPECT_KEY_CLOSE_QUOTE"
        elif state == "EXPECT_KEY_CLOSE_QUOTE":
            state = "EXPECT_COLON"
        elif state == "EXPECT_COLON":
            state = "EXPECT_VALUE"
        elif state == "EXPECT_VALUE":
            if current_arg_type in ("float", "int"):
                state = "EXPECT_VALUE_CONTINUE"
            elif current_arg_type == "str":
                state = "EXPECT_VALUE_CONTENT"
                value_content_count = 0
            else:
                state = "EXPECT_COMMA"
        elif state == "EXPECT_VALUE_CONTINUE":
            if best_token_str in (",", "}"):
                if best_token_str == ",":
                    state = "EXPECT_KEY_OPEN_QUOTE"
                    current_arg = args_to_process.pop(0)
                    current_arg_type = function.args_types[current_arg]
                    current_arg_tokens = encode(model, current_arg)
                else:
                    state = "END"
        elif state == "EXPECT_VALUE_CONTENT":
            value_content_count += 1
            if best_token_str == '"':
                state = "EXPECT_COMMA"
                value_content_count = 0
            elif value_content_count >= MAX_VALUE_CONTENT_TOKENS:
                state = "EXPECT_VALUE_CLOSE_QUOTE"
                value_content_count = 0
        elif state == "EXPECT_VALUE_CLOSE_QUOTE":
            state = "EXPECT_COMMA"
        elif state == "EXPECT_COMMA":
            if args_to_process:
                state = "EXPECT_KEY_OPEN_QUOTE"
                current_arg = args_to_process.pop(0)
                current_arg_type = function.args_types[current_arg]
                current_arg_tokens = encode(model, current_arg)
            else:
                state = "END"

    json_str = "".join(
        clean_token(vocab_by_id.get(t, "")) for t in generated_token_ids
    )
    start = json_str.find("{")
    end = json_str.rfind("}") + 1
    json_str = json_str[start:end]
    result: dict[str, Any] = json.loads(json_str)
    return result
