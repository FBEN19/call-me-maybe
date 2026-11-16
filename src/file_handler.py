import json
import os
from pydantic import ValidationError
from .models import FunctionDefinition, Prompt, FunctionCall


def load_functions(path: str) -> list[FunctionDefinition]:
    """Load function definitions from a JSON file.

    Args:
        path: Path to the JSON file containing function definitions.

    Returns:
        A list of FunctionDefinition objects, or an empty list on error.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            return [FunctionDefinition(**item) for item in data]
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: invalid JSON in file: {path}")
        return []
    except ValidationError:
        print(f"Error: data does not match expected schema in: {path}")
        return []


def load_prompts(path: str) -> list[Prompt]:
    """Load prompts from a JSON file.

    Args:
        path: Path to the JSON file containing prompt objects.

    Returns:
        A list of Prompt objects, or an empty list on error.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            return [Prompt(**item) for item in data]
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: invalid JSON in file: {path}")
        return []
    except ValidationError:
        print(f"Error: data does not match expected schema in: {path}")
        return []


def write_results(results: list[FunctionCall], path: str) -> None:
    """Write function call results to a JSON output file.

    Args:
        results: List of FunctionCall objects to serialize.
        path: Destination file path for the JSON output.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        all_data = [result.model_dump() for result in results]
        with open(path, "w") as f:
            json.dump(all_data, f, indent=2)
    except OSError as e:
        print(f"Error writing file {path}: {e}")
