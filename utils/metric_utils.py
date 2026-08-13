def calculate_token_costs(prompt_tokens: int, completion_tokens: int) -> float:
    """Local Ollama execution incurs no transaction costs."""
    return 0.0