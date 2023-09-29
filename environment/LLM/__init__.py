from .rater import LLMRater
from .exllama import LLMExllama
from .openai_api import OpenAIModelAPI

SUPPORTED_MODELS = [
    "TheBloke/Llama-2-7b-Chat-GPTQ",  # use via exllama, on 8gb gpu
    "TheBloke/Llama-2-13B-chat-GPTQ",  # use via exllama, on 24gb gpu
    "TheBloke/vicuna-7B-v1.3-GPTQ",  # use via exllama, on 8gb gpu
    "TheBloke/vicuna-13b-v1.3.0-GPTQ",  # use via exllama, on 24gb gpu
    "TheBloke/vicuna-33B-GPTQ",  # use via exllama, on 24gb gpu
    "TheBloke/vicuna-7B-v1.5-GPTQ",  # use via exllama, on 8gb gpu
    "TheBloke/vicuna-13B-v1.5-GPTQ",  # use via exllama, on 24gb gpu
    "TheBloke/Llama-2-70B-chat-GPTQ",  # use via exllama, on 80gb gpu
    "gpt-3.5-turbo-0613",  # use via openai api
    "gpt-4-0613",  # use via openai api
]


def load_LLM(name):
    if not name in SUPPORTED_MODELS:
        raise ValueError(f"Model {name} not supported.")
    elif "gpt-3.5-turbo-0613" == name or "gpt-4-0613" in name:
        return OpenAIModelAPI(name)
    elif "GPTQ" in name:
        return LLMExllama(name)
