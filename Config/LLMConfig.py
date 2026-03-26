import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("")

class LLMConfig:
    """Configuration for Large Language Models"""

    # LLM Provider
    # Options: "openai", "anthropic", "deepseek", "llama" (local vLLM)
    # PROVIDER = "deepseek"
    PROVIDER = "llama"


    # Model Selection
    OPENAI_MODEL = "gpt-4o"                    # or "gpt-4", "gpt-3.5-turbo"
    ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
    DEEPSEEK_MODEL = "deepseek-chat"
    HUGGINGFACE_MODEL = "meta-llama/Llama-2-70b-chat-hf"
    # LLAMA_MODEL = "models/Llama-3.1-8B-Instruct-AWQ"
    # LLAMA_MODEL_NAME = "llama-3.1-8b-awq"  # Simple name for file paths (no slashes)
    # --- [Added] Llama 3.1 70B AWQ-INT4 ---
    # LLAMA_MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    # LLAMA_MODEL_NAME = "llama-3.1-70b-awq-int4"  # Simple name for file paths (no slashes)
    # --- [End Added] ---
    # --- [Modified] Llama 3.1 8B AWQ-INT4 for answering ---
    LLAMA_MODEL = "/home/lhz/code/model/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    LLAMA_MODEL_NAME = "llama-3.1-8b-awq-int4"  # Simple name for file paths (no slashes)
    # --- [End Modified] ---


    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


    # API Base URLs
    OPENAI_BASE_URL = "https://api.openai.com/v1"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
    # LLAMA_BASE_URL = "http://localhost:8000/v1"
    # --- [Modified] Llama 3.1 8B on port 8001 for answering ---
    LLAMA_BASE_URL = "http://localhost:8001/v1"
    # --- [End Modified] ---


    # Generation Parameters
    TEMPERATURE = 0.7                          # Randomness (0.0-2.0)
    MAX_TOKENS = 1000                          # Maximum tokens in response
    TOP_P = 0.9                                # Nucleus sampling


    # API Request Settings
    TIMEOUT = 60                               # Request timeout in seconds
    MAX_RETRIES = 3                            # Number of retries on failure
    RETRY_DELAY = 2                            # Delay between retries in seconds


    # Response Format
    STREAM = False                             # Stream responses
    RESPONSE_FORMAT = "text"                   # "text" or "json"


    # Helper Methods
    @staticmethod
    def get_model_config():
        """Get current model configuration based on PROVIDER

        Returns:
            Dictionary with keys: model, api_key, base_url
        """
        provider = LLMConfig.PROVIDER.lower()

        if provider == "openai":
            return {
                "model": LLMConfig.OPENAI_MODEL,
                "api_key": LLMConfig.OPENAI_API_KEY,
                "base_url": LLMConfig.OPENAI_BASE_URL
            }
        elif provider == "deepseek":
            return {
                "model": LLMConfig.DEEPSEEK_MODEL,
                "api_key": LLMConfig.DEEPSEEK_API_KEY,
                "base_url": LLMConfig.DEEPSEEK_BASE_URL
            }
        elif provider == "anthropic":
            return {
                "model": LLMConfig.ANTHROPIC_MODEL,
                "api_key": LLMConfig.ANTHROPIC_API_KEY,
                "base_url": LLMConfig.ANTHROPIC_BASE_URL
            }
        elif provider == "llama":
            return {
                "model": LLMConfig.LLAMA_MODEL,
                "model_name": LLMConfig.LLAMA_MODEL_NAME,  # For file paths
                "api_key": "not-needed",  # vLLM doesn't require API key
                "base_url": LLMConfig.LLAMA_BASE_URL
            }
        else:
            raise ValueError(f"Unsupported provider: {LLMConfig.PROVIDER}")


# Usage example
if __name__ == "__main__":
    print(f"Provider: {LLMConfig.PROVIDER}")
    print(f"Model: {LLMConfig.OPENAI_MODEL}")
    print(f"Temperature: {LLMConfig.TEMPERATURE}")
    print(f"Max Tokens: {LLMConfig.MAX_TOKENS}")
