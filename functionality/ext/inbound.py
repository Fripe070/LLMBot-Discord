import datetime
import io
import logging
import re
import time

import discord
from yarl import URL

from .defs import CheckupContext

__all__ = (
    "process_incoming",
)

from ..apis import caption

_logger = logging.getLogger(__name__)


def get_attachment_tag(kind: str, cap: str) -> str:
    return f"<file type={kind}>{cap}</file>"


async def process_incoming(message: discord.Message, *, ctx: CheckupContext) -> str:
    content = message.system_content.strip()
    assert message.guild is not None, "Message must be in a guild"

    emoji_map = {str(emoji): f":{emoji.name}:" for emoji in message.guild.emojis}
    for emoji_str, emoji_rep in emoji_map.items():
        content = content.replace(emoji_str, emoji_rep)

    # Replace mentions with their index in the author_indexes
    for match_full, match_snowflake in re.findall(r"(<@!?([0-9]+)>)", content):
        mapped_snowflake = ctx.author_indexes.setdefault(int(match_snowflake), len(ctx.author_indexes))
        content = content.replace(match_full, f"@{mapped_snowflake}")

    content = await process_media(message, content, ctx=ctx)

    return content

async def process_media(message: discord.Message, content: str, *, ctx: CheckupContext) -> str:
    attachment_start_time = time.process_time()
    content = content.rstrip() + " "
    for attachment in message.attachments:
        try:
            processed_caption = await process_attachment(attachment, ctx=ctx)
        except Exception as error:
            _logger.warning(
                f"Error processing attachment {attachment.filename!r} {attachment.url!r}: {error!r}", exc_info=error
            )
            continue
        if processed_caption is None:
            _logger.debug(f"Attachment {attachment.filename} could not be processed. Skipping.")
            continue
        _logger.debug(f"Processed attachment {attachment.filename!r} {attachment.url}:\n{processed_caption!r}")
        content += processed_caption + " "
    content = content.strip()
    if message.attachments:
        attachment_duration = time.process_time() - attachment_start_time
        _logger.debug(
            f"Processed {len(message.attachments)} attachments in {attachment_duration:.2f} seconds. "
            f"{attachment_duration / len(message.attachments):.2f} seconds per attachment."
        )

    embed_start_time = time.process_time()
    content = content.rstrip() + " "
    for embed in message.embeds:
        if embed.type == "image":
            if not embed.url:
                _logger.debug(f"Embed {embed} is of type 'image' but has no URL. Skipping processing.")
                continue
            img_tag = await process_image(URL(embed.url), ctx=ctx)
            if img_tag is None:
                _logger.debug(f"Image embed {embed.url!r} could not be processed. Skipping.")
                continue
            if embed.url and embed.url in content:
                content = content.replace(embed.url, img_tag)
            else:
                content += img_tag + " "
        else:
            embed_tag = await process_generic_embed(embed, ctx=ctx)
            if embed_tag is None:
                _logger.debug(f"Embed {embed} could not be processed. Skipping.")
                continue
            if embed.url and embed.url in content:
                content = content.replace(embed.url, embed_tag)
            else:
                content += embed_tag + " "
    content = content.rstrip()
    if message.embeds:
        embed_duration = time.process_time() - embed_start_time
        _logger.debug(
            f"Processed {len(message.embeds)} embeds in {embed_duration:.2f} seconds. "
            f"{embed_duration / len(message.embeds):.2f} seconds per embed."
        )

    if message.poll:
        try:
            poll_tag = await process_poll(message.poll, ctx=ctx)
        except Exception as error:
            _logger.warning(f"Error processing poll {message.poll!r}: {error!r}", exc_info=error)
        else:
            content = (content.rstrip() + " " + poll_tag).strip()

    return content


async def process_poll(poll: discord.Poll, *, ctx: CheckupContext) -> str:
    poll_attrs: dict[str, str] = {
        "multiple_choice": "true" if poll.multiple else "false",
    }
    if poll.duration < datetime.timedelta(days=1):
        poll_attrs["duration"] = f"{poll.duration.total_seconds()/60/60:.0f}h"
    else:
        poll_attrs["duration"] = f"{poll.duration.total_seconds()/60/60/24:.0f}d"

    result = "<poll"
    for attr, value in poll_attrs.items():
        result += f' {attr}="{value}"'
    result += ">"
    result += f"{poll.question}"
    for answer in poll.answers:
        result += f"<answer>{answer.text}</answer>"
    result += "</poll>"
    return result


async def process_generic_embed(embed: discord.Embed, *, ctx: CheckupContext) -> str | None:
    if embed.type not in {"rich"}: # TODO: Article? No change needs to be made, but it might be best treated as a URL
        _logger.debug(f"Encountered unsupported embed type {embed.type!r}. Skipping processing.")
        return None

    summary: str = embed.description or ""
    if embed.title:
        if summary:
            summary = f"# {embed.title}\n{summary}"
        else:
            summary = embed.title
    summary = summary.rstrip()
    for img in filter(bool, (embed.thumbnail, embed.image)):
        if not img.url:
            _logger.debug(f"Embed {embed.url!r} has an image without a URL. Skipping processing.")
            continue
        img_tag = await process_image(URL(img.url), ctx=ctx)
        if img_tag is None:
            _logger.debug(f"Image in embed {embed.url!r} could not be processed. Skipping.")
            continue
        summary += "\n" + img_tag
    summary = summary.strip()

    return f"<embed>{summary}</embed>"


async def process_attachment(attachment: discord.Attachment, *, ctx: CheckupContext) -> str | None:
    unrecognized_attachment_tag: str = get_attachment_tag(
        "unknown",
        f'Unsupported attachment type for file "{attachment.filename}"'
    )

    if attachment.size > ctx.config.max_attachment_size_mb * 1024 * 1024:
        _logger.debug(
            f"Attachment {attachment.filename} is too large ({attachment.size / (1024 * 1024):.2f} MB). "
            f"Skipping processing."
        )
        return None

    _logger.debug(f"Attachment {attachment.filename!r} has content type {attachment.content_type!r}.")

    image_cts = {re.compile(r"image/(png|jpeg|webp)")}
    supported_cts: set[re.Pattern] = image_cts

    matching_ct: re.Pattern | None = None
    for ct in supported_cts:
        if ct.match(attachment.content_type):
            matching_ct = ct
            break
    if matching_ct is None:
        _logger.debug(
            f"Attachment {attachment.filename} has unsupported content type {attachment.content_type!r}. "
            "Skipping processing."
        )
        return unrecognized_attachment_tag

    file = io.BytesIO()
    await attachment.save(file, seek_begin=True)

    if matching_ct in image_cts:
        return await process_image(file, ctx=ctx)

    _logger.warning(f'Attachment of type "{attachment.content_type}" was recognized but not supported.')
    return unrecognized_attachment_tag


async def process_image(image: io.BytesIO | URL, *, ctx: CheckupContext) -> str | None:
    if isinstance(image, io.BytesIO):
        file = image
    elif isinstance(image, URL):
        async with ctx.apis.session.get(image) as response:
            if response.status != 200:
                _logger.warning(f"Failed to fetch image from {image!r}: {response.status} {response.reason}")
                return None
            file = io.BytesIO(await response.read())
    else:
        raise ValueError(f"Unsupported image type: {type(image)}. Expected io.BytesIO or URL.")

    caption_text = await caption.generate_image_caption(
        file,
        model=ctx.config.models.chat,
        ollama_client=ctx.apis.ollama,
    )
    tag = get_attachment_tag("image", caption_text)
    return tag