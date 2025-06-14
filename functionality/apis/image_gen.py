import asyncio
import base64
import datetime
import io
import logging
from collections.abc import Collection
from typing import Any

import aiohttp
from yarl import URL

__all__ = (
    "HordeSession",
    "generate_image",
)


_logger = logging.getLogger(__name__)


class HordeSession:
    def __init__(
        self,
        api_key: str,
        session: aiohttp.ClientSession,
        *,
        base_url: URL | str = "https://aihorde.net/api/",
    ) -> None:
        self.session = session
        self.horde_url = URL(base_url)
        self.headers = {
            "apikey": api_key,
            "Client-Agent": "LLMBot:0.1:github.com/Fripe070/LLMBot-Discord"
        }


async def _throwing_horde_request(
    session: HordeSession,
    method: str,
    url: URL,
    **kwargs: Any,
) -> dict[str, Any]:
    async with session.session.request(method=method, url=url, headers=session.headers, **kwargs) as response:
        try:
            data = await response.json()
        except Exception as error:
            raise RuntimeError(f"Failed to parse horde response: {error}") from error
        if not response.ok or "rc" in data:
            raise RuntimeError(
                f"Failed to make horde request: "
                f"{data.get("rc", "Unknown")} (HTTP {response.status}) {data.get("message", "N/A")}"
            )
        for warning in data.get("warnings", []):
            _logger.warning(f"Horde request warning {warning.get("code", "Unknown")}: {warning.get("message", "N/A")}")
    return data


async def cancel_generation(session: HordeSession, generation_id: str) -> None:
    await _throwing_horde_request(session, "DELETE", session.horde_url / f"v2/generate/status/{generation_id}")


async def wait_for_completion(
    session: HordeSession,
    generation_id: str,
    *,
    timeout: datetime.timedelta = datetime.timedelta(minutes=10),
) -> None:
    start_time = datetime.datetime.now()
    while timeout > (datetime.datetime.now() - start_time):
        data = await _throwing_horde_request(
            session, "GET",
            session.horde_url / f"v2/generate/status/{generation_id}" % {"id": generation_id},
        )
        if data["faulted"]:
            raise RuntimeError(f"Generation {generation_id} failed: {data.get('message', 'No message provided')}")
        if data["done"]:
            _logger.debug(f"Generation {generation_id} completed successfully.")
            return
        _logger.debug(f"Generation {generation_id} is still in progress. Eta: {data["wait_time"]}")
        await asyncio.sleep(10)
        
    await cancel_generation(session, generation_id)
    raise TimeoutError(f"Generation {generation_id} timed out after {timeout}.")


async def generate_image(
    prompt: str,
    session: HordeSession,
    *,
    models: Collection[str] | None = None,
    censor_nsfw: bool = True,
    width: int = 512,
    height: int = 512,
    style: str | None = None,
    extra_params: dict[str, Any] | None = None,
    timeout: datetime.timedelta = datetime.timedelta(minutes=10),
) -> io.BytesIO | None:
    opts: dict[str, Any] = {
        "prompt": prompt,
        "nsfw": not censor_nsfw,
        "censor_nsfw": censor_nsfw,
        "params": {
            "width": width,
            "height": height,
            **(extra_params or {}),
        },
    }
    if models:
        opts["models"] = list(models)
    if style:
        opts["style"] = style

    data = await _throwing_horde_request(
        session,
        "POST", session.horde_url / "v2/generate/async",
        json=opts,
    )
    generation_id: str = data["id"]
    for warning in data.get("warnings", []):
        if warning.get("code") in ("NoAvailableWorker",):
            _logger.warning(f"Generation {generation_id} has no available workers. Cancelling.")
            await cancel_generation(session, generation_id)
            return None
        
    _logger.debug(f"Image generation started with ID: {generation_id}")

    check = await wait_for_completion(session, generation_id, timeout=timeout)
    

    data = await _throwing_horde_request(session, "GET", session.horde_url / f"v2/generate/status/{generation_id}")
    generations: list = data.get("generations", [])
    if not generations:
        raise RuntimeError(f"No generations returned for generation: {generation_id}")
    generation = generations[0]
    metadata: list[dict[str, str]] = generation.get("metadata", [])
    for meta in metadata:
        if meta["type"] == "censorship":
            return None  # Censored image, return None
    image: str = generation["img"]
    
    if image.startswith("http"):
        async with session.session.get(image) as response:
            if not response.ok:
                raise RuntimeError(f"Failed to fetch image from r2: {response.status} {response.reason} ({image})")
            image_data = await response.read()
            return io.BytesIO(image_data)
    else:
        return io.BytesIO(base64.b64decode(image))







