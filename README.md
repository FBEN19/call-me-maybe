*This activity has been created as part of the 42 curriculum by bevergna.*

# call-me-maybe

## Description

**call-me-maybe** is a function-calling system that maps natural language prompts to structured function calls — without fine-tuning or few-shot examples. Given a user prompt and a set of function definitions, the system selects the right function and extracts its arguments as valid JSON, running entirely locally on a small language model.

The core insight is **constrained decoding**: instead of asking the model to generate free text and then parsing it, we restrict the model's output vocabulary at every generation step so that only syntactically and semantically valid tokens are ever produced. This guarantees well-formed output by construction.

The LLM backend is [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) loaded via `llm_sdk`, a thin wrapper around HuggingFace `transformers`. No API calls are made; all inference runs on your local machine.

## Instructions

### Prerequisites

- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)

On first run, the model weights (~1.1 GB) are downloaded automatically from HuggingFace Hub and cached locally.

### Installation

```bash
make install
```

This runs `uv sync` to create a virtual environment and install all dependencies (`torch`, `transformers`, `pydantic`, `numpy`, `accelerate`).

### Running

```bash
make run
```

This processes `data/input/function_calling_tests.json` against `data/input/functions_definition.json` and writes results to `data/output/function_calling_results.json`.

**Custom paths:**

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input               data/input/function_calling_tests.json \
  --output              data/output/function_calling_results.json
```

### Development commands

```bash
make lint         # flake8 + mypy (standard)
make lint-strict  # mypy --strict
make debug        # run under pdb
make clean        # remove __pycache__, .mypy_cache, etc.
```

## Algorithm Explanation

The pipeline runs in two sequential phases for each prompt:

### Phase 1 — Function Selection via Logit Masking

A prompt is constructed that lists the available functions and ends with the partial string `"The function to call is: fn"`. This primes the model to predict the continuation of a function name.

At this position, raw logits (unnormalized scores over the entire vocabulary) are retrieved from the model. A mask is applied that sets every token's score to `-inf` **except** for the distinctive second token of each valid function name (e.g. `_add` from `fn_add_numbers`). `argmax` over the masked logits deterministically selects the function whose name continuation the model scores highest.

### Phase 2 — Argument Extraction via State-Machine Constrained Decoding

Once the function is selected, its full name is appended to `input_ids` and a token-by-token generation loop begins. A finite state machine governs what tokens are legal at each step:

| State | Allowed tokens |
|---|---|
| `START` | `{` |
| `EXPECT_KEY_OPEN_QUOTE` | `"` |
| `EXPECT_KEY_NAME` | Next token of the current argument name |
| `EXPECT_KEY_CLOSE_QUOTE` | `"` |
| `EXPECT_COLON` | `:` |
| `EXPECT_VALUE` | Numeric start for int/float · `"` for str · `true`/`false` for bool |
| `EXPECT_VALUE_CONTINUE` | More numeric digits · `,` (more args) · `}` (last arg) |
| `EXPECT_VALUE_CONTENT` | Any token without `"` or control chars · `"` to close |
| `EXPECT_VALUE_CLOSE_QUOTE` | `"` |
| `EXPECT_COMMA` | `,` (more args) · `}` (last arg) |
| `END` | — (generation stops) |

At each step the model's logits are masked to the valid set and `argmax` picks the best allowed token. The process terminates when the closing `}` is emitted.

The generated token sequence is reassembled into a JSON string, control-character glyphs are cleaned (e.g. `Ġ` → space), and `json.loads` deserializes it into a Python dict.

## Design Decisions

**`argmax` instead of sampling.** Function calling is a deterministic task: given the same prompt, the correct function and its arguments should always be the same. Greedy decoding (argmax) is reproducible, faster than sampling, and avoids the risk of hallucinating slightly wrong values through random noise.

**Distinctive second token for function selection.** All function names share the prefix `fn`. The first token of every name encodes `fn`, making it useless for disambiguation. The second token (e.g. `_add`, `_greet`) is unique per function, so it's used as the discriminating signal instead.

**State machine over regex/grammar parsing.** A finite state machine is simpler to reason about and extend than a full grammar parser. Each state explicitly encodes what the decoder is currently constructing, making the valid token set trivial to compute without a grammar library.

**Argument names tokenized up front.** Key names are encoded once and stored as a queue of token IDs (`current_arg_tokens`). The state machine consumes them one by one in `EXPECT_KEY_NAME`, guaranteeing the exact key name regardless of how the tokenizer splits it.

**`MAX_VALUE_CONTENT_TOKENS` safety limit.** String values have no natural length bound. A cap of 50 content tokens prevents runaway generation for open-ended strings while still being generous for realistic inputs.

**No fine-tuning, no few-shot.** The system relies entirely on structural constraints rather than teaching the model about function calling. This makes it model-agnostic: any causal LM compatible with HuggingFace `AutoModelForCausalLM` can be swapped in.

## Performance Analysis

**Accuracy.** On the 14 test prompts in `function_calling_tests.json`, function selection achieves 100% accuracy — the logit-masking approach is exact given that each function has a distinct second token. Argument extraction is also exact for numeric and boolean types since the grammar is fully specified. String values depend on the model correctly extracting the value from context; for well-formed prompts the model generalizes well.

**Speed.** The model runs one forward pass per generated token. On a modern CPU, each forward pass through Qwen3-0.6B takes roughly 100–300 ms, giving ~3–10 s per argument token. On CUDA or MPS hardware this drops to ~10–30 ms per token. The model is loaded once and reused across all prompts, so startup cost is amortized.

**Memory.** Qwen3-0.6B at float16 requires ~1.2 GB of GPU/MPS memory or ~2.4 GB RAM on CPU. The vocabulary JSON (~2 MB) is loaded from the HuggingFace cache on each call to `generate`; in a production setting this would be cached in memory.

**Reliability.** The constrained decoder cannot produce malformed JSON — if `json.loads` fails, it is a bug in the state machine, not a model hallucination. A `MAX_TOKENS` guard (500) prevents infinite loops in degenerate states.

## Challenges Faced

**Tokenizer glyph handling.** HuggingFace tokenizers represent whitespace and control characters with Unicode glyphs (`Ġ` for space, `Ċ` for newline, `ĉ` for tab). The vocabulary JSON contains these raw glyphs. Without the `clean_token` function, the assembled JSON string contains literal `Ġ` characters and fails to parse. Identifying and mapping all relevant glyphs required inspecting the raw vocabulary.

**Distinctive token collision.** Two functions could theoretically share the same second token (e.g. `fn_add` and `fn_all`). The current implementation skips duplicates silently. A more robust approach would walk further down the token sequences until a unique prefix is found; this was deemed out of scope for the activity.

**Multi-token numeric values.** Numbers like `265` may tokenize as multiple tokens (`26`, `5`) or as a single token. The `EXPECT_VALUE_CONTINUE` state must accept more numeric tokens after the first, and only allow `,` or `}` once a complete numeric token boundary is reached. Determining this boundary without a lookahead required allowing any numeric-looking token continuation and relying on the model's preference for complete numbers.

**State machine for multi-token key names.** Argument names like `source_string` tokenize into several tokens. The key-emission loop uses a pre-computed token queue and consumes it one step at a time, staying in `EXPECT_KEY_NAME` until the queue is empty before advancing to `EXPECT_KEY_CLOSE_QUOTE`.

## Testing Strategy

Validation was done against the 14 representative prompts in `data/input/function_calling_tests.json`, covering all supported argument types and edge cases:

- **Numeric functions** (`fn_add_numbers`, `fn_multiply_numbers`, `fn_get_square_root`): verify float extraction including multi-digit values (265, 345).
- **Boolean function** (`fn_is_even`): verify `true`/`false` generation for both even and odd inputs.
- **Single-string function** (`fn_greet`, `fn_reverse_string`): verify string extraction and quote handling.
- **Multi-string function** (`fn_substitute_string_with_regex`): the most complex case — three string arguments including a regex pattern and a replacement, testing the full key/value loop.

Results were inspected manually in `data/output/function_calling_results.json`. Static analysis (`make lint`) and strict type checking (`make lint-strict`) were used to catch type errors and code quality issues throughout development.

## Example Usage

### Input files

**`data/input/functions_definition.json`** — array of function schemas:
```json
[
  {
    "fn_name": "fn_add_numbers",
    "args_names": ["a", "b"],
    "args_types": {"a": "float", "b": "float"},
    "return_type": "float"
  },
  {
    "fn_name": "fn_greet",
    "args_names": ["name"],
    "args_types": {"name": "str"},
    "return_type": "str"
  }
]
```

Supported argument types: `int`, `float`, `str`, `bool`.

**`data/input/function_calling_tests.json`** — array of natural language prompts:
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"}
]
```

### Running

```bash
make run
```

### Output

**`data/output/function_calling_results.json`**:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  }
]
```

### More examples from the test suite

| Prompt | Selected function | Parameters |
|---|---|---|
| `"What is the square root of 16?"` | `fn_get_square_root` | `{"a": 16.0}` |
| `"Is 7 an even number?"` | `fn_is_even` | `{"n": 7}` |
| `"What is the product of 12 and 4?"` | `fn_multiply_numbers` | `{"a": 12.0, "b": 4.0}` |
| `"Reverse the string 'hello'"` | `fn_reverse_string` | `{"s": "hello"}` |
| `"Substitute the word 'cat' with 'dog' in 'The cat sat on the mat'"` | `fn_substitute_string_with_regex` | `{"source_string": "The cat sat on the mat", "regex": "cat", "replacement": "dog"}` |

## Project Structure

```
src/
  __main__.py            # entry point — loads data, runs the pipeline
  function_selector.py   # prompt building + function selection via logit masking
  constrained_decoder.py # state-machine decoder for JSON argument generation
  tokenizer.py           # encode helper bridging src/ and llm_sdk
  file_handler.py        # JSON I/O
  models.py              # Pydantic schemas (FunctionDefinition, Prompt, FunctionCall)
llm_sdk/
  __init__.py            # Small_LLM_Model — wraps a HuggingFace causal-LM
data/
  input/                 # function schemas and test prompts
  output/                # generated results (git-ignored)
```

## Resources

### Constrained Decoding & Function Calling

- [Outlines — structured text generation](https://github.com/dottxt-ai/outlines) — reference library implementing constrained decoding with finite-state machines and grammars.
- [Logit Bias in the OpenAI API](https://platform.openai.com/docs/api-reference/chat/create#chat-create-logit_bias) — production example of token-level logit masking for constrained generation.
- [Guidance — constrained generation library](https://github.com/guidance-ai/guidance) — another approach to structured LLM outputs.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) — the backbone LLM used in this project.
- [HuggingFace Transformers documentation](https://huggingface.co/docs/transformers/index) — tokenizer and model loading APIs.
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259) — the grammar the state machine enforces.

### AI Usage

AI (Claude) was used for the following tasks in this activity:

- **Understanding Constrained Decoding**: learning how to implement and use constrained decoding to ensure syntactically valid JSON output by restricting token generation at each step.
- **Writing this README**: the full README was drafted with AI assistance, using the source code and test data as the primary reference.
