import argparse
from llm_sdk import Small_LLM_Model
from .file_handler import load_functions, load_prompts, write_results
from .function_selector import select_function


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for input/output file paths.

    Returns:
        Parsed namespace with functions_definition, input, and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Function calling tool using constrained decoding"
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json"
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load inputs, run function selection, write results."""
    args = parse_args()
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    if not functions or not prompts:
        print("Error: could not load input files, aborting.")
        return

    model = Small_LLM_Model()
    results = []
    for prompt in prompts:
        result = select_function(prompt, functions, model)
        results.append(result)

    write_results(results, args.output)
    print(f"Done: {len(results)} results written to {args.output}")


if __name__ == "__main__":
    main()
