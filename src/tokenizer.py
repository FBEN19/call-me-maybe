from llm_sdk import Small_LLM_Model


def encode(model: Small_LLM_Model, text: str) -> list[int]:
    return model._encode(text)[0].tolist()
