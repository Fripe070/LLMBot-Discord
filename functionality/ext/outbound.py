import asyncio
import collections
import io
import logging
import math
import random
import re
import dataclasses

import discord
from rapidfuzz import fuzz
from yarl import URL

from .defs import CheckupContext
from ..apis import searching, image_gen
from ..util.formatting import URL_REGEX, URL_LINK_REGEX, URL_REGEX_NO_EMBED

__all__ = (
    "process_outgoing",
    "DiscordResponse",
)

_logger = logging.getLogger(__name__)

@dataclasses.dataclass(kw_only=True)
class DiscordResponse:
    content: str
    attachments: list[discord.File] = dataclasses.field(default_factory=list)
    embeds: list[discord.Embed] = dataclasses.field(default_factory=list)


async def process_outgoing(content: str, *, ctx: CheckupContext) -> DiscordResponse | None:
    # Almost guaranteed to be an invalid response
    if re.search(r"[msgid:[0-9]+]", content):
        _logger.debug(f"Invalid response content detected: {content!r}. Skipping processing.")
        return None

    if "<file type=unknown>" in content:
        _logger.debug("Content contains unknown file type tag. Skipping processing.")
        return None
    
    if content.startswith(tuple(map(str, range(10)))):  # STOP SENDING NUMBERS :sob:
        _logger.debug(f"Content starts with a number: {content!r}. Skipping processing.")
        return None

    character_counts = collections.Counter(content)
    frequencies = ((c / len(content)) for c in character_counts.values())
    entropy = -sum(freq * math.log(freq, 2) for freq in frequencies)
    if entropy > 5:
        _logger.debug(f"Content has too high entropy: {entropy:.2f}. Skipping processing.")
        return None

    # REMEMBER TO UPDATE THE ACTUAL GENERATION IF CHANGING THIS REGEX!
    if not (
        len(re.findall(r"<file type=image>|<image>", flags=re.IGNORECASE, string=content))
        == 
        len(re.findall(r"</(?:file>|image)>", flags=re.IGNORECASE, string=content))
    ):
        _logger.debug("Mismatched amount of image start and end tags.")
        return None

    # Replace emotes with their Discord representation
    emoji_map = {f":{emoji.name}:": emoji for emoji in ctx.channel.guild.emojis}
    for emoji_str, emoji in emoji_map.items():
        content = content.replace(emoji_str, str(emoji))

    inverse_author_indexes = {v: k for k, v in ctx.author_indexes.items()}
    for match_full, match_index in (
        *re.findall(r"(@#?([0-9]+)\b)", content),
        *re.findall(r"((?:@|\b)User ?#?([0-9]+)\b)", content, flags=re.IGNORECASE),
    ):
        match_index_int = int(match_index)
        if 0 <= match_index_int >= len(inverse_author_indexes):
            _logger.debug(f"Invalid author index {match_index_int} in content: {content!r}. Skipping processing.")
            return None
        mapped_snowflake = inverse_author_indexes.get(match_index_int, None)
        if mapped_snowflake is None:
            _logger.debug(
                f"Author index {match_index_int} not found in author indexes: {ctx.author_indexes}. Skipping processing."
            )
            return None
        content = content.replace(match_full, f"<@{mapped_snowflake}>")

    links: list[re.Match[str]] = [
        *re.finditer(URL_REGEX, content),
        *re.finditer(URL_REGEX_NO_EMBED, content),
        *re.finditer(URL_LINK_REGEX, content),
    ]
    urls: set[str] = {match["url"] for match in links}
    processed_urls: list[str | None | BaseException] = await asyncio.gather(
        *(process_url(url, ctx=ctx) for url in urls),
        return_exceptions=True,
    )
    for original_url, processed_url in zip(urls, processed_urls):
        if processed_url is None:
            _logger.debug(f"URL {original_url} could not be processed. Skipping.")
            continue
        if not isinstance(processed_url, str):
            _logger.debug(f"Error processing URL {original_url}: {processed_url!r}")
            continue
        _logger.debug(f"{original_url} -> {processed_url}")
        content = content.replace(original_url, processed_url)

    # Check if content is too similar to a previously sent response
    similar_enough: int = 0
    for cached in ctx.previously_sent_cache:
        if fuzz.partial_ratio(content, cached) / 100 >= ctx.channel_config.repeat_prevention.threshold:
            similar_enough += 1
    if similar_enough >= ctx.channel_config.repeat_prevention.max_messages:
        _logger.debug(f"Response content is too similar to previously sent content. Skipping this response.")
        return None

    # It LOVES sending incrementing numbers.
    def is_num(string: str) -> bool:
        return all(c in "0123456789" for c in string if c not in "\\`*_[]()<>~ \n")
    if is_num(content) and any(is_num(cached) for cached in ctx.previously_sent_cache):
        _logger.debug(f"Response content is a number and similar to previously sent content. Skipping this response.")
        return None

    response: DiscordResponse = DiscordResponse(content=content)
    del content  # I have accidentally tried writing to this so many times
    
    # REMEMBER TO UPDATE THE PREVIOUSLY PERFORMED CHECK IF CHANGING THIS REGEX!
    image_attempts: list[re.Match[str]] = [match for match in re.finditer(
        r"(?:<file type=image>|<image>)(.+?)(?:</file>|</image>)",
        flags=re.IGNORECASE | re.DOTALL,
        string=response.content,
    )]
    img_gen_tasks: list[asyncio.Task] = []
    for i, attempted_image in enumerate(image_attempts):
        attempted_image_prompt: str = attempted_image.group(1).strip()
        _logger.debug(f"Attempting to generate image for prompt: {attempted_image_prompt!r}")
        img_gen_tasks.append(asyncio.create_task(image_gen.generate_image(
            session=ctx.apis.horde,
            prompt=attempted_image_prompt,
            style=ctx.config.horde_image_style,
        )))
        if i+1 != len(image_attempts):
            await asyncio.sleep(10)
            
    generated_images: list[io.BytesIO | None | BaseException] = await asyncio.gather(*img_gen_tasks, return_exceptions=True)
    for i, (attempt, generation) in enumerate(zip(image_attempts, generated_images)):
        if isinstance(generation, BaseException):
            _logger.error(f"Error generating image: {generation} for prompt: {attempt.group(1)!r}")
            continue
        if generation is None:
            _logger.debug(f"Image generation failed for prompt: {attempt.group(1)!r}")
            continue
        _logger.debug(f"Image generated successfully for prompt: {attempt.group(1)!r}")
        response.content = response.content.replace(attempt.group(0), "")
        response.attachments.append(discord.File(
            fp=generation,
            filename=f"image_{i}.webp",
            description=f'AI-generated image based on the prompt: "{attempt.group(1)}"',
        ))
        
    # TODO: Send polls

    embed_attempts: list[re.Match[str]] = [match for match in re.finditer(
        r'<embed(?:\s+colou?r="(?P<colour>.+?)")?>(?P<content>.+?)</embed>',
        flags=re.IGNORECASE | re.DOTALL,
        string=response.content,
    )]
    for embed_attempt in embed_attempts:
        embed_content: str = embed_attempt["content"].strip()
        _logger.debug(f"Attempting to create embed from content: {embed_content!r}")
        if not embed_content.startswith("#"):
            title, description = None, embed_content
        else:
            title, _, description = embed_content.partition("\n")
            title = title.lstrip("#")
        colour: discord.Colour | None = None
        if embed_attempt["colour"]:
            try:
                colour = discord.Colour.from_str(embed_attempt["colour"])
            except ValueError:
                _logger.debug(f"Embed colour could not be converted: {embed_attempt['colour']!r}. Skipping.")

        response.content = response.content.replace(embed_attempt.group(0), "")
        response.embeds.append(discord.Embed(
            title=title,
            description=description,
            colour=colour,
        ))

    # TODO: React to messages

    return response

async def process_url(url: str, *, ctx: CheckupContext) -> str | None:
    try:
        parsed_url: URL = URL(url)
    except ValueError:
        return None
    if parsed_url.scheme not in {"http", "https"}:
        return None
    if not parsed_url.host:
        return None

    if ctx.config.google_api_key and (
        parsed_url.host.endswith(("tenor.com", "giphy.com")) or parsed_url.suffix in {".gif", ".webp"}
    ):
        tenor_query = await searching.generate_search_query(
            "\n".join(msg.content for msg in ctx.history),
            prompt=searching.GIF_PROMPT,
            model=ctx.config.models.chat,
            ollama_client=ctx.apis.ollama,
        )
        _logger.debug(f"Searching Tenor for: {tenor_query}")
        search_results = await searching.search_tenor(
            tenor_query,
            api_key=ctx.config.google_api_key,
            limit=3,
        )
        if search_results:
            return str(random.choice(search_results).url)
        else:
            _logger.warning(f"No Tenor results found for query: {tenor_query}")

    # Any engines that use normal search queries are handled below
    query = await searching.generate_search_query(
        "\n".join(msg.content for msg in ctx.history),
        model=ctx.config.models.chat,
        ollama_client=ctx.apis.ollama,
    )
    _logger.debug(f"Generated query for URL {url}: {query}")

    if parsed_url.host.endswith(("youtube.com", "youtu.be")):
        search_results = await searching.search_searx(query, api_url=ctx.config.searx_url, engines=["youtube"])
        if not search_results:
            raise ValueError(f"No YouTube results found for query: {query}")
        return str(search_results[0].url)

    # We just search for the query in general search engines
    if ctx.config.searx_url:
        search_results = await searching.search_searx(query, api_url=ctx.config.searx_url)
        if not search_results:
            raise ValueError(f"No search results found for query: {query}")
        return str(search_results[0].url)

    _logger.debug(f"Skipping URL {url} as it went unhandled.")
    return None

