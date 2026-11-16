from pydantic import BaseModel
from typing import Any


class FunctionDefinition(BaseModel):
    """Represents a callable function definition loaded from JSON.

    Attributes:
        fn_name: The name of the function.
        args_names: Ordered list of argument names.
        args_types: Mapping from argument name to its type string.
        return_type: The return type of the function as a string.
    """

    fn_name: str
    args_names: list[str]
    args_types: dict[str, str]
    return_type: str


class Prompt(BaseModel):
    """A natural language prompt to be processed.

    Attributes:
        prompt: The raw natural language request string.
    """

    prompt: str


class FunctionCall(BaseModel):
    """The result of mapping a prompt to a function call.

    Attributes:
        prompt: The original natural language request.
        name: The name of the selected function.
        parameters: The extracted arguments mapped to their values.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
