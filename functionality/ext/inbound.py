import asyncio
import io
import re
import logging

import discord

from .defs import CheckupContext

__all__ = (
    "process_incoming",
)

from ..apis import caption

_logger = logging.getLogger(__name__)


async def process_incoming(message: discord.Message, *, ctx: CheckupContext) -> str:
    content = message.content.strip()
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
    processed_attachments: list[str | None | BaseException] = await asyncio.gather(
        *(process_attachment(attachment, ctx=ctx) for attachment in message.attachments),
        return_exceptions=True,
    )
    content = content.rstrip() + " "
    for attachment, caption_text in zip(message.attachments, processed_attachments):
        if isinstance(caption_text, BaseException):
            _logger.error(f"Error processing attachment {attachment.filename}: {caption_text!r}")
            continue
        if caption_text is None:
            _logger.debug(f"Attachment {attachment.filename} could not be processed. Skipping.")
            continue
        _logger.debug(f"Processed attachment {attachment.filename}:\n{caption_text!r}")
        content += caption_text + " "
    content = content.strip()
    
    return content

async def process_attachment(attachment: discord.Attachment, *, ctx: CheckupContext) -> str | None:
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
        return None

    file = io.BytesIO()
    await attachment.save(file, seek_begin=True)

    def get_attachment_tag(kind: str, cap: str) -> str:
        return f"<file type={kind}>{cap}</file>"

    if matching_ct in image_cts:
        caption_text = await caption.generate_image_caption(
            file,
            model=ctx.config.models.chat,
            ollama_client=ctx.apis.ollama,
        )
        tag = get_attachment_tag("image", caption_text)
        return tag
    
    return None
