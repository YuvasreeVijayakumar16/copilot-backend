from openai import AzureOpenAI
import os
import logging

logger = logging.getLogger(__name__)

_client = None

def get_openai_client() -> AzureOpenAI:
    """
    Returns a singleton Azure OpenAI client.
    """

    global _client

    if _client is not None:
        return _client

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("CLOUD_PROVIDER_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    # Validate required settings
    if not endpoint:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT environment variable is missing."
        )

    if not api_key:
        raise ValueError(
            "CLOUD_PROVIDER_OPENAI_API_KEY environment variable is missing."
        )

    if not deployment:
        raise ValueError(
            "AZURE_OPENAI_DEPLOYMENT environment variable is missing."
        )

    logger.info("Initializing Azure OpenAI client")
    logger.info(f"Azure Endpoint: {endpoint}")
    logger.info(f"Azure Deployment: {deployment}")

    _client = AzureOpenAI(
        api_key=api_key,
        api_version="2025-04-01-preview",
        azure_endpoint=endpoint,
    )

    return _client


def get_openai_model(default: str = "gpt-4o") -> str:
    """
    Returns Azure deployment name.
    """
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if deployment:
        return deployment

    return os.getenv("AUTOGEN_MODEL", default)