import logging

import ollama as ollama_api

__all__ = (
    "ensure_downloaded",
    "models_match",
    "generify_model_name",
)

_logger = logging.getLogger(__name__)


def generify_model_name(model: str) -> str:
    return model.removesuffix(":latest")


def models_match(a: ollama_api.ListResponse.Model | str, b: ollama_api.ListResponse.Model | str) -> bool:
    name_a: str | None = a.model if isinstance(a, ollama_api.ListResponse.Model) else a
    name_b: str | None = b.model if isinstance(b, ollama_api.ListResponse.Model) else b
    if not name_a or not name_b:
        return False
    return generify_model_name(name_a) == generify_model_name(name_b)


async def ensure_downloaded(model_name: str, *, ollama_client: ollama_api.AsyncClient) -> None:
    if any(models_match(model, model_name) for model in (await ollama_client.list()).models):
        _logger.debug(f'Language model "{model_name}" already found in ollama. Skipping download.')
    else:
        _logger.info(f'Downloading model "{model_name}"')
        prev_status: str | None = None
        async for download_step in await ollama_client.pull(model_name, stream=True):
            if download_step.status == prev_status:
                continue
            _logger.debug(f"Ollama downloading {model_name!r}: {download_step.status}")
            prev_status = download_step.status
