import asyncio
import collections
import dataclasses
import datetime
import logging
import random
import re
import time
from collections.abc import Sequence

import discord
import ollama as ollama_api
from discord.ext import commands
from yarl import URL

from bot.bot import LLMBot
from bot.config import DUMMY_SNOWFLAKE, ConfigError, ChannelConfig
from .apis import llm, searching
from .util.formatting import chop_string, URL_REGEX, URL_LINK_REGEX, URL_REGEX_NO_EMBED

type DiscordSnowflake = int
type ChannelID = DiscordSnowflake
type MessageID = DiscordSnowflake


_logger = logging.getLogger(__name__)

MAX_GENERATION_RETRIES: int = 15

@dataclasses.dataclass(kw_only=True)
class CheckupContext:
    config: ChannelConfig
    channel: discord.TextChannel
    history: Sequence[discord.Message]
    author_indexes: dict[DiscordSnowflake, int] = dataclasses.field(default_factory=dict)
    is_response_to_latest: bool


class AIBotFunctionality(commands.Cog, name="Bot Functionality"):
    def __init__(self, bot: LLMBot) -> None:
        self.bot = bot
        self.bg_task: asyncio.Task | None = None
        self.is_cog_ready: bool = False

        self.ollama = ollama_api.AsyncClient(host=bot.config.ollama_url)

        self.schedule: dict[discord.TextChannel, datetime.datetime] = {}
        self.channel_histories: dict[ChannelID, collections.deque[discord.Message]] = {}

        for channel in self.bot.config.channels.values():
            if channel.channel_id == DUMMY_SNOWFLAKE:
                raise ConfigError(
                    f"A channel is using the default (invalid) channel ID of {DUMMY_SNOWFLAKE}. "
                    "Please update your configuration."
                )

    async def cog_load(self) -> None:
        def runner_done_callback(task: asyncio.Task[None]) -> None:
            try:
                task.result()  # Propagate any exceptions raised in the background task
            except asyncio.CancelledError:
                _logger.debug("Background runner task was cancelled.")

        self.bg_task = self.bot.loop.create_task(self.background_runner())
        self.bg_task.add_done_callback(runner_done_callback)

    async def cog_unload(self) -> None:
        if self.bg_task is not None and not self.bg_task.done():
            self.bg_task.cancel()

    async def background_runner(self) -> None:
        await llm.ensure_downloaded(self.bot.config.models.text, client=self.ollama)
        await llm.ensure_downloaded(self.bot.config.models.chat, client=self.ollama)

        await self.bot.wait_until_ready()

        # Initialise mappings and schedule initial checkups
        for channel_config in self.bot.config.channels.values():
            channel_obj = (
                self.bot.get_channel(channel_config.channel_id)
                or await self.bot.fetch_channel(channel_config.channel_id)
            )
            if not isinstance(channel_obj, discord.TextChannel):
                raise ConfigError(
                    f"Channel {channel_config.channel_id} is not a valid text channel. "
                    "Please check your configuration."
                )
            self.schedule[channel_obj] = datetime.datetime.now() # ASAP
            self.channel_histories[channel_obj.id] = collections.deque(maxlen=channel_config.max_history)
            _logger.debug(f"Initialising channel history for #{channel_obj.name} with {channel_config.max_history} messages.")
            async for msg in channel_obj.history(before=datetime.datetime.now(), limit=channel_config.max_history):
                self.channel_histories[channel_obj.id].appendleft(msg)

        self.is_cog_ready = True

        min_checkup_interval: datetime.timedelta = min(
            channel_config.checkup_interval - channel_config.checkup_variance
            for channel_config in self.bot.config.channels.values()
        )
        _logger.debug(f"Checkup loop starting with a minimum interval of {min_checkup_interval.total_seconds():.2f} seconds.")
        while True:
            start_time: float = time.perf_counter()
            await self.check_scheduled_channels()
            elapsed_time: float = time.perf_counter() - start_time
            # We don't want to hog the event loop
            await asyncio.sleep(max(0.0, min_checkup_interval.total_seconds() - elapsed_time))

    async def check_scheduled_channels(self):
        for channel_obj, scheduled_for in self.schedule.items():
            if scheduled_for > datetime.datetime.now():
                continue
            channel_config = self.bot.config.channels[channel_obj.id]

            history: Sequence[discord.Message] = self.channel_histories[channel_obj.id]
            if len(history) <= 0:
                _logger.debug(f"No messages in channel {channel_obj.name} to process. Skipping checkup.")
                continue
            assert len(history) <= channel_config.max_history, (
                f"Channel {channel_obj.name} has more messages ({len(history)}) "
                f"than the configured maximum ({channel_config.max_history})."
            )

            if not self.is_too_active(history, channel_config):
                await self.channel_checkup(
                    channel=channel_obj,
                    history=history,
                )
            else:
                _logger.debug(f"Channel {channel_obj.name} is too active. Skipping checkup.")
            # Reschedule
            self.schedule[channel_obj] = datetime.datetime.now() + channel_config.random_interval()

    def is_too_active(self, history: Sequence[discord.Message], channel_config: ChannelConfig) -> bool:
        """Check if the bot is too active in the channel."""
        assert self.bot.user is not None, "Not logged in. Cannot check bot activity."
        if channel_config.activity_limit.max_bot_messages <= 0:
            return False
        bot_sighting_count: int = 0
        for i in range(-channel_config.activity_limit.window_size, 0):
            if history[i].author.id == self.bot.user.id:
                bot_sighting_count += 1
                if bot_sighting_count >= channel_config.activity_limit.max_bot_messages:
                    return True
        return False

    @commands.Cog.listener("on_message")
    async def on_message(self, message: discord.Message) -> None:
        if not self.is_cog_ready:
            return

        if message.channel.id not in self.bot.config.channels:
            return
        assert isinstance(message.channel, discord.TextChannel), (
            f"Message channel {message.channel.id} is not a text channel. "
            f"THis is the result of illegal config state that should have failed a check earlier. "
        )

        # TODO: Temporary. Will exclude images which I dont rly want. Solution until embeds/attachments are handled
        if not message.content.strip():
            return

        self.channel_histories[message.channel.id].append(message)

        if self.bot.user in message.mentions and not message.author.id == self.bot.user.id:
            _logger.debug(f"Message {message.id} in #{message.channel.name} mentions the bot. Performing checkup.")
            channel_config = self.bot.config.channels[message.channel.id]

            history: collections.deque[discord.Message] = collections.deque(
                maxlen=channel_config.history_when_responding,
            )
            for msg in reversed(self.channel_histories[message.channel.id]):
                if msg.id < message.id and len(history) < channel_config.history_when_responding:
                    history.appendleft(msg)
            if len(history) < channel_config.history_when_responding:
                # Fill out the rest with a fetch
                async for msg in message.channel.history(
                    before=history[0],
                    limit=channel_config.history_when_responding - len(history),
                ):
                    history.appendleft(msg)

            await self.channel_checkup(
                channel=message.channel,
                history=history,
                is_response_to_latest=True,
            )

    async def channel_checkup(
        self,
        *,
        channel: discord.TextChannel,
        history: Sequence[discord.Message],
        is_response_to_latest: bool = False,
    ) -> None:
        assert all(
            msg1.id < msg2.id
            for msg1, msg2 in zip(history, tuple(history)[1:])
        ), "History messages are out of order. Later messages should have higher IDs than earlier ones."

        ctx = CheckupContext(
            config=self.bot.config.channels[channel.id],
            channel=channel,
            history=history,
            is_response_to_latest=is_response_to_latest,
        )

        prompt: str = (
            "The following is the chat history of a Discord channel. Markdown is supported.\n"
            "Mention format: `@UserID`. For example: @1 for User #1.\n"
        )
        if channel.guild.emojis:
            emojis = " ".join(f":{emoji.name}:" for emoji in channel.guild.emojis)
            prompt += f"Chatroom emotes: {emojis}\n"

        prompt += "\n" + "\n".join(await self.gather_prompt_history(ctx)) + "\n"

        assert self.bot.user is not None, "Bot user is not set."
        prompt += f"[msgid:{len(history)}] User #"
        if ctx.config.talk_as_bot:
            prompt += str(ctx.author_indexes.setdefault(self.bot.user.id, len(ctx.author_indexes)))
        else:
            # TODO: Attempt to have the LLM predict the author rather than using a random one
            prompt += str(random.choice(tuple(ctx.author_indexes.values())))

        if is_response_to_latest:
            prompt += f" (Replying to [msgid:{len(history) - 1}]): "

        _logger.debug(f"Working prompt for channel #{channel.name}:\n{prompt}")

        response_content, reply_to = await self.generate_response(prompt, ctx=ctx)

        _logger.debug(f"Sending message in #{channel.name}:\n{response_content}")

        allowed_mentions = discord.AllowedMentions(replied_user=True, users=True, everyone=False, roles=False)
        # TODO: Do something more intelligent than just chopping the string
        if reply_to is None:
            await channel.send(
                content=chop_string(response_content, 2000),
                allowed_mentions=allowed_mentions,
            )
        else:
            await reply_to.reply(
                content=chop_string(response_content, 2000),
                allowed_mentions=allowed_mentions,
            )

    async def gather_prompt_history(self, ctx: CheckupContext) -> Sequence[str]:
        processed_history: list[str] = []
        for message_index, message in enumerate(ctx.history):
            author_index = ctx.author_indexes.setdefault(message.webhook_id or message.author.id, len(ctx.author_indexes))
            working_string = f"[msgid:{message_index}] User #{author_index}"
            if message.reference is not None and message.reference.message_id is not None:
                for i in range(len(ctx.history)):
                    if ctx.history[i].id == message.reference.message_id:
                        working_string += f" (replying to [msgid:{i}])"
                        break
            working_string += ": "
            working_string += await self.process_incoming(message, ctx=ctx)
            processed_history.append(working_string)
        return processed_history

    async def generate_response(self, prompt: str, *, ctx: CheckupContext) -> tuple[str, discord.Message | None]:
        # We simply retry until we get a valid response from the LLM. It can be finicky sometimes.
        for attempt_index in range(MAX_GENERATION_RETRIES):
            result = await self.ollama.generate(
                model=self.bot.config.models.text,
                prompt=prompt,
                options=ollama_api.Options(
                    stop=["\n[msgid:"],
                    num_predict=500,
                ),
                keep_alive=5,
            )
            _logger.debug(f"LLM response (attempt {attempt_index + 1}): {result.response!r}")

            content: str | None
            replying_to_index: int | None = None
            if ctx.is_response_to_latest:
                replying_to_index = len(ctx.history) - 1
                content = result.response.strip()
            else:
                reply_prefix, reply_suffix = "(Replying to [msgid:", "]): "
                if not result.response.lstrip().startswith(reply_prefix):
                    content = result.response.strip().removeprefix(": ")
                else:
                    reply_i_str, _, content = result.response.lstrip().removeprefix(reply_prefix).partition(reply_suffix)
                    try:
                        replying_to_index = int(reply_i_str.strip())
                    except ValueError:
                        continue
                    if replying_to_index < 0 or replying_to_index >= len(ctx.history):
                        continue

            content = await self.process_outgoing(content, ctx=ctx)
            if content is None:
                continue
            return content, ctx.history[replying_to_index] if replying_to_index is not None else None

        raise RuntimeError(f"Failed to generate a valid response after {MAX_GENERATION_RETRIES} attempts.")

    async def process_incoming(
        self,
        message: discord.Message,
        *,
        ctx: CheckupContext,
    ) -> str:
        # TODO: Surface more message data (e.g. attachments, embeds, images)
        content = message.content.strip()
        assert message.guild is not None, "Message must be in a guild"

        emoji_map = {str(emoji): f":{emoji.name}:" for emoji in message.guild.emojis}
        for emoji_str, emoji_rep in emoji_map.items():
            content = content.replace(emoji_str, emoji_rep)

        # Replace mentions with their index in the author_indexes
        for match_full, match_snowflake in re.findall(r"(<@!?([0-9]+)>)", content):
            mapped_snowflake = ctx.author_indexes.setdefault(int(match_snowflake), len(ctx.author_indexes))
            content = content.replace(match_full, f"@{mapped_snowflake}")

        return content

    async def process_outgoing(
        self,
        content: str,
        *,
        ctx: CheckupContext,
    ) -> str | None:
        # TODO: *Technically* we should have a check for if the content is in a codeblock

        # Almost guaranteed to be an invalid response
        if re.search(r"[msgid:[0-9]+]", content):
            return None

        # Replace emotes with their Discord representation
        emoji_map = {f":{emoji.name}:": emoji for emoji in ctx.channel.guild.emojis}
        for emoji_str, emoji in emoji_map.items():
            content = content.replace(emoji_str, str(emoji))

        inverse_author_indexes = {v: k for k, v in ctx.author_indexes.items()}
        for match_full, match_index in (
            *re.findall(r"(@([0-9]+)\b)", content),
            *re.findall(r"(\bUser ?([0-9]+)\b)", content, flags=re.IGNORECASE),
        ):
            match_index_int = int(match_index)
            if 0 <= match_index_int >= len(inverse_author_indexes):
                return None
            mapped_snowflake = inverse_author_indexes.get(match_index_int, None)
            if mapped_snowflake is None:
                return None
            content = content.replace(match_full, f"<@{mapped_snowflake}>")

        links: list[re.Match[str]] = [
            *re.finditer(URL_REGEX, content),
            *re.finditer(URL_REGEX_NO_EMBED, content),
            *re.finditer(URL_LINK_REGEX, content),
        ]
        urls: set[str] = {match["url"] for match in links}
        processed_urls: list[str | None | BaseException] = await asyncio.gather(
            *(self.process_url(url, ctx=ctx) for url in urls),
            return_exceptions=True,
        )
        for original_url, processed_url in zip(urls, processed_urls):
            if processed_url is None:
                continue
            if not isinstance(processed_url, str):
                _logger.debug(f"Error processing URL {original_url}: {processed_url!r}")
                continue
            _logger.debug(f"{original_url} -> {processed_url}")
            content = content.replace(original_url, processed_url)

        # TODO: Images


        return content

    async def process_url(
        self,
        url: str,
        *,
        ctx: CheckupContext,
    ) -> str | None:
        try:
            parsed_url: URL = URL(url)
        except ValueError:
            return None
        if parsed_url.scheme not in {"http", "https"}:
            return None
        if not parsed_url.host:
            return None

        if self.bot.config.google_api_key and (
            parsed_url.host.endswith(("tenor.com", "giphy.com"))
            or parsed_url.suffix in {".gif", ".webp"}
        ):
            tenor_query = await searching.generate_search_query(
                "\n".join(msg.content for msg in ctx.history),
                prompt=searching.GIF_PROMPT,
                model=self.bot.config.models.chat,
                ollama_api=self.ollama,
            )
            _logger.debug(f"Searching Tenor for: {tenor_query}")
            search_results = await searching.search_tenor(
                tenor_query,
                api_key=self.bot.config.google_api_key,
                limit=3,
            )
            if search_results:
                return str(random.choice(search_results).url)
            else:
                _logger.warning(f"No Tenor results found for query: {tenor_query}")

        # Any engines that use normal search queries are handled below
        query = await searching.generate_search_query(
            "\n".join(msg.content for msg in ctx.history),
            model=self.bot.config.models.chat,
            ollama_api=self.ollama,
        )
        _logger.debug(f"Generated query for URL {url}: {query}")

        if parsed_url.host.endswith(("youtube.com", "youtu.be")):
            search_results = await searching.search_searx(
                query,
                api_url=self.bot.config.searx_url,
                engines=["youtube"]
            )
            if not search_results:
                raise ValueError(f"No YouTube results found for query: {query}")
            return str(search_results[0].url)

        # We just search for the query in general search engines
        if self.bot.config.searx_url:
            search_results = await searching.search_searx(query, api_url=self.bot.config.searx_url)
            if not search_results:
                raise ValueError(f"No search results found for query: {query}")
            return str(search_results[0].url)
        
        _logger.debug(f"Skipping URL {url} as it went unhandled.")
        return None


async def setup(bot: LLMBot) -> None:
    await bot.add_cog(AIBotFunctionality(bot))
