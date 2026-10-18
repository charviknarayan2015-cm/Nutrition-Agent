"""
Shared LLM + Embedding client factory for all agents.
Initializes IBM watsonx.ai WatsonxLLM once and reuses across agents.
"""
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from nutrition_agent.config import (
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
    PRIMARY_MODEL,
    GENERATION_PARAMS,
)


def get_credentials() -> Credentials:
    return Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)


def get_llm_model() -> ModelInference:
    """Return a reusable ModelInference instance."""
    credentials = get_credentials()
    params = {
        GenParams.MAX_NEW_TOKENS:      GENERATION_PARAMS["max_new_tokens"],
        GenParams.MIN_NEW_TOKENS:      GENERATION_PARAMS["min_new_tokens"],
        GenParams.TEMPERATURE:         GENERATION_PARAMS["temperature"],
        GenParams.TOP_P:               GENERATION_PARAMS["top_p"],
        GenParams.REPETITION_PENALTY:  GENERATION_PARAMS["repetition_penalty"],
    }
    return ModelInference(
        model_id=PRIMARY_MODEL,
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID,
        params=params,
    )


# Module-level singleton — initialised lazily
_llm_model: ModelInference | None = None


def get_shared_llm() -> ModelInference:
    global _llm_model
    if _llm_model is None:
        _llm_model = get_llm_model()
    return _llm_model


def generate_text(prompt: str, max_tokens: int = 1024) -> str:
    """Convenience wrapper: send a prompt and return the generated text."""
    model = get_shared_llm()
    response = model.generate_text(prompt=prompt)
    return response.strip() if isinstance(response, str) else str(response).strip()
