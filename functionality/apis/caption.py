import logging
import base64
import asyncio
import io
from PIL import Image

import aiohttp
import ollama as ollama_api

__all__ = ("generate_image_caption",)

_logger = logging.getLogger(__name__)


def resize_img(image_data: bytes, max_size: tuple[int, int] = (896, 896)) -> bytes:
    img = Image.open(io.BytesIO(image_data))

    ratio = min(max_size[0] / img.width, max_size[1] / img.height)
    resized_img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.Resampling.LANCZOS)

    padded_img = Image.new("RGBA", max_size, (0, 0, 0, 0))
    padded_img.paste(resized_img, ((max_size[0] - resized_img.width) // 2, (max_size[1] - resized_img.height) // 2))

    output = io.BytesIO()
    padded_img.save(output, format="PNG")
    return output.getvalue()


async def generate_image_caption(
    img_url: str,
    *,
    model: str = "gemma3:4b-it-qat",
    client: ollama_api.AsyncClient = ollama_api.AsyncClient(),
) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(img_url) as res:
                res.raise_for_status()
                img_bytes = await res.read()

        img_b64 = base64.b64encode(await asyncio.to_thread(resize_img, img_bytes)).decode("utf-8")
        res = await client.generate(
            model=model,
            prompt="Describe this image in a short sentence, only include the description and text within the image in response",
            images=[img_b64],
            options=ollama_api.Options(
                num_predict=150,
            ),
        )
        return res.response.strip()

    except Exception as e:
        _logger.error(f"Error generating image caption: {e}")
        raise
