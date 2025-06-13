import asyncio
import io
import re
import logging
import time

import discord

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

    # TODO: Surface more message data
    # TODO: Embeds
    # TODO: Polls
    # TODO: Audio (files and voice messages)
    attachment_start_time = time.process_time()
    content = content.rstrip() + " "
    for attachment in message.attachments:
        try:
            processed_caption = await process_attachment(attachment, ctx=ctx)
        except Exception as error:
            _logger.warning(f"Error processing attachment {attachment.filename!r} {attachment.url!r}: {error!r}", exc_info=error)
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

    return content

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
        caption_text = await caption.generate_image_caption(
            file,
            model=ctx.config.models.chat,
            ollama_client=ctx.apis.ollama,
        )
        tag = get_attachment_tag("image", caption_text)
        return tag

    _logger.warning(f'Attachment of type "{attachment.content_type}" was recognized but not supported.')
    return unrecognized_attachment_tag
