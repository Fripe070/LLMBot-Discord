import asyncio
import io
import logging

import ollama as ollama_api
from PIL import Image

from .llm import generify_model_name

__all__ = ("generate_image_caption", "supports_vision")

_logger = logging.getLogger(__name__)

IMAGE_CAPTION_PROMPT = (
    "Describe this image in detail but in a single line. "
    "Only include the image description and any text that appears within the image in your response."
)

_vision_capabilities: dict[str, bool] = {}

async def supports_vision(model: str, ollama_client: ollama_api.AsyncClient) -> bool:
    model_name = generify_model_name(model)
    if model_name not in _vision_capabilities:
        model_info = await ollama_client.show(model)
        _vision_capabilities[model_name] = "vision" in (model_info.capabilities or [])
    return _vision_capabilities[model_name]


def resize_within(
    image: Image.Image,
    *,
    box: tuple[int, int] = (896, 896),
    background: tuple[float, float ,float, float] = (0, 0, 0, 0),
) -> Image.Image:
    result = Image.new("RGBA", box, background)
    
    ratio: float = min(box[0] / image.width, box[1] / image.height)
    resized_img = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.Resampling.LANCZOS)

    result.paste(resized_img, ((box[0] - resized_img.width) // 2, (box[1] - resized_img.height) // 2))
    return result

def clamp_size(
    image: Image.Image,
    *,
    max_size: tuple[int, int] = (896, 896),
) -> Image.Image:
    if image.width <= max_size[0] and image.height <= max_size[1]:
        return image
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def _process_image(img: Image.Image | io.BytesIO) -> bytes:
    if isinstance(img, io.BytesIO):
        img = Image.open(img)
    output = io.BytesIO()
    # resize_within(img).save(output, format="PNG")
    clamp_size(img).save(output, format="PNG")
    return output.getvalue()


async def generate_image_caption(
    image: bytes | io.BytesIO | Image.Image,
    *,
    model: str,
    ollama_client: ollama_api.AsyncClient,
    prompt: str = IMAGE_CAPTION_PROMPT,
) -> str:
    if not await supports_vision(model, ollama_client):
        raise ValueError(f"Model {model} does not support vision capabilities.")

    if isinstance(image, bytes):
        image = io.BytesIO(image)
    if not isinstance(image, (io.BytesIO, Image.Image)):
        raise ValueError("Input must be a bytes, BytesIO, or a PIL Image object.")

    loop = asyncio.get_running_loop()
    processed = await loop.run_in_executor(None, _process_image, image)

    result = await ollama_client.generate(
        model=model,
        prompt=prompt,
        images=[ollama_api.Image(value=processed)],
        options=ollama_api.Options(
            stop=["\n"],
            num_predict=150,
        ),
    )
    return result.response.strip()
