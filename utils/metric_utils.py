from config import config

def calculate_token_costs(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates granular transaction cost estimates for Gemini 2.5 Flash."""
    input_cost = (prompt_tokens / 1_000_000) * config.PRICE_PER_1M_INPUT_TOKENS
    output_cost = (completion_tokens / 1_000_000) * config.PRICE_PER_1M_OUTPUT_TOKENS
    return round(input_cost + output_cost, 6)