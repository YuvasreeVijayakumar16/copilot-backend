import json
import logging
import os
from datetime import datetime, timezone
from uuid import uuid4

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
except Exception:  # pragma: no cover - lets the app run if optional package is absent
    BlobServiceClient = None
    ContentSettings = None


logger = logging.getLogger(__name__)

_blob_service_client = None


def _get_blob_service_client():
    global _blob_service_client
    if _blob_service_client is not None:
        return _blob_service_client

    connection_string = (
        os.getenv("AZURE_PROMPT_LOG_CONNECTION_STRING")
        or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )
    if not connection_string or BlobServiceClient is None:
        return None

    _blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    return _blob_service_client


def _json_default(value):
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def log_prompt_to_blob(
    *,
    source: str,
    model: str,
    messages: list,
    metadata: dict | None = None,
    response_text: str | None = None,
) -> None:
    """Persist an LLM prompt/response trace to Azure Blob when configured.

    Set AZURE_PROMPT_LOG_CONNECTION_STRING or AZURE_STORAGE_CONNECTION_STRING.
    Optional:
      AZURE_PROMPT_LOG_CONTAINER defaults to prompt-logs
      AZURE_PROMPT_LOG_PREFIX defaults to prompts
      AZURE_PROMPT_LOG_ENABLED=false disables logging
    """
    if os.getenv("AZURE_PROMPT_LOG_ENABLED", "true").lower() in {"0", "false", "no"}:
        return

    client = _get_blob_service_client()
    if client is None:
        return

    container_name = os.getenv("AZURE_PROMPT_LOG_CONTAINER", "prompt-logs")
    prefix = os.getenv("AZURE_PROMPT_LOG_PREFIX", "prompts").strip("/")
    now = datetime.now(timezone.utc)

    payload = {
        "id": str(uuid4()),
        "timestamp_utc": now.isoformat(),
        "source": source,
        "model": model,
        "messages": messages,
        "metadata": metadata or {},
        "response_text": response_text,
    }

    blob_name = (
        f"{prefix}/{now.strftime('%Y/%m/%d/%H')}/"
        f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{payload['id']}.json"
    )

    try:
        container = client.get_container_client(container_name)
        try:
            container.create_container()
        except Exception:
            pass

        data = json.dumps(payload, ensure_ascii=False, default=_json_default, indent=2)
        container.upload_blob(
            name=blob_name,
            data=data,
            overwrite=False,
            content_settings=ContentSettings(content_type="application/json")
            if ContentSettings
            else None,
        )
    except Exception:
        logger.exception("Failed to upload prompt log to Azure Blob")
