import asyncio
import collections
import datetime
import logging
import random
import time
from collections.abc import Sequence

import aiohttp
import discord
import ollama as ollama_api
from discord.ext import commands

from bot.bot import LLMBot
from bot.config import DUMMY_SNOWFLAKE, ConfigError, ChannelConfig
from .defs import APISet, ChannelID, CheckupContext, MAX_GENERATION_RETRIES
from .inbound import process_incoming
from .outbound import process_outgoing, DiscordResponse
from ..apis import llm, image_gen
from ..util.formatting import chop_string

_logger = logging.getLogger(__name__)

CACHED_RESPONSES_MAX_LENGTH: int = 5

class AIBotFunctionality(commands.Cog, name="Bot Functionality"):
    def __init__(self, bot: LLMBot) -> None:
        self.bot = bot
        self.bg_task: asyncio.Task | None = None
        self.is_cog_ready: bool = False

        self.apis: APISet | None = None

        self.schedule: dict[discord.TextChannel, datetime.datetime] = {}
        self.channel_histories: dict[ChannelID, collections.deque[discord.Message]] = {}
        self.channel_responses_sent: dict[ChannelID, collections.deque[str]] = {}

        for channel in self.bot.config.channels.values():
            if channel.channel_id == DUMMY_SNOWFLAKE:
                raise ConfigError(
                    f"A channel is using the default (invalid) channel ID of {DUMMY_SNOWFLAKE}. "
                    "Please update your configuration."
                )

    async def cog_load(self) -> None:
        self.apis = APISet(
            ollama=ollama_api.AsyncClient(host=self.bot.config.ollama_url),
            session=aiohttp.ClientSession(),
            horde=image_gen.HordeSession(
                api_key=self.bot.config.horde_key,
                session=aiohttp.ClientSession(),
                base_url=self.bot.config.horde_url,
            ),
        )
        
        def runner_done_callback(task: asyncio.Task[None]) -> None:
            try:
                task.result()  # Propagate any exceptions raised in the background task
            except asyncio.CancelledError:
                _logger.info("Background runner task was cancelled.")

        self.bg_task = self.bot.loop.create_task(self.background_runner())
        self.bg_task.add_done_callback(runner_done_callback)

    async def cog_unload(self) -> None:
        if self.bg_task is not None and not self.bg_task.done():
            self.bg_task.cancel()
        if self.apis is not None and not self.apis.session.closed:
            await self.apis.session.close()
        if self.apis is not None and not self.apis.horde.session.closed:
            await self.apis.horde.session.close()

    async def background_runner(self) -> None:
        """Background task that runs the checkup loop."""
        _logger.info("Starting background runner for AI bot functionality.")
        if self.apis is None:
            raise RuntimeError("APIs are not initialized. This is likely the result of invalid cog loading.")
        await llm.ensure_downloaded(self.bot.config.models.text, ollama_client=self.apis.ollama)
        await llm.ensure_downloaded(self.bot.config.models.chat, ollama_client=self.apis.ollama)

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
            self.channel_histories[channel_obj.id] = collections.deque(maxlen=channel_config.history.max)
            _logger.debug(f"Initialising channel history for #{channel_obj.name} with {channel_config.history.max} messages.")
            async for msg in channel_obj.history(before=datetime.datetime.now(), limit=channel_config.history.max):
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

            history: Sequence[discord.Message] = tuple(
                self.channel_histories[channel_obj.id]
            )[-channel_config.history.limit:]
            if len(history) <= 0:
                _logger.debug(f"No messages in channel {channel_obj.name} to process. Skipping checkup.")
                continue

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
            if -i >= len(history):
                continue
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

        self.channel_histories[message.channel.id].append(message)

        if self.bot.user in message.mentions and not message.author.id == self.bot.user.id:
            _logger.debug(f"Message {message.id} in #{message.channel.name} mentions the bot. Performing checkup.")
            channel_config = self.bot.config.channels[message.channel.id]

            history: collections.deque[discord.Message] = collections.deque(
                maxlen=channel_config.history.limit_responding,
            )
            for msg in reversed(self.channel_histories[message.channel.id]):
                if msg.id < message.id and len(history) < channel_config.history.limit_responding - 1:
                    history.appendleft(msg)
            if len(history) < channel_config.history.limit_responding:
                # Fill out the rest with a fetch
                async for msg in message.channel.history(
                    before=history[0],
                    limit=channel_config.history.limit_responding - len(history) - 1,
                ):
                    history.appendleft(msg)
            history.append(message)

            assert all(msg.id <= message.id for msg in history), "History contains messages newer than the triggering message."
            assert history[-1].id == message.id, "Latest message in history is not the triggering message."

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
        assert all(msg1.id < msg2.id for msg1, msg2 in zip(history, tuple(history)[1:])), (
            "History messages are out of order. Later messages should have higher IDs than earlier ones."
        )
        
        channel_config = self.bot.config.channels[channel.id]
        assert self.apis, "APIs are not initialized. This is likely the result of invalid cog loading."
        ctx = CheckupContext(
            config=self.bot.config,
            channel_config=channel_config,
            apis=self.apis,
            channel=channel,
            history=tuple(history),
            previously_sent_cache=self.channel_responses_sent.setdefault(
                channel.id, collections.deque(maxlen=channel_config.repeat_prevention.window_size)),
            is_response_to_latest=is_response_to_latest,
        )

        prompt: str = (
            "The following is the chat history of a Discord channel. Markdown is supported.\n"
            "Mention format: `@UserID`. For example: @1 for User #1.\n"
        )
        if channel.guild.emojis:
            emojis = " ".join(f":{emoji.name}:" for emoji in channel.guild.emojis)
            prompt += f"Chatroom emotes: {emojis}\n"

        prompt_history: Sequence[str] = await self.gather_prompt_history(ctx)
        prompt += "\n" + "\n".join(prompt_history) + "\n"

        appropriate_history_length = (
            ctx.channel_config.history.limit_responding
            if is_response_to_latest
            else ctx.channel_config.history.limit
        )
        assert len(prompt_history) == len(history), (
            f"Prompt history length {len(prompt_history)} does not match history length {len(history)}."
        )
        assert len(prompt_history) <= appropriate_history_length, (
            f"History length {len(history)} exceeds the maximum allowed length of {appropriate_history_length}."
        )
        assert len(prompt_history) == appropriate_history_length, (
            f"History length does not match the expected length. {len(prompt_history)} != {appropriate_history_length}. "
        )

        assert self.bot.user is not None, "Bot user is not set."
        prompt += f"[msgid:{len(history)}] User #"
        if ctx.channel_config.talk_as_bot or not ctx.author_indexes.values():
            prompt += str(ctx.author_indexes.setdefault(self.bot.user.id, len(ctx.author_indexes)))
        else:
            # TODO: Attempt to have the LLM predict the author rather than using a random one
            mimicked_id: int = random.choice(tuple(ctx.author_indexes.values()))
            prompt = prompt.replace(f"@{ctx.author_indexes[self.bot.user.id]}", f"@{mimicked_id}") # TODO: Flawed. Is there a better way?
            prompt += str(mimicked_id)

        if is_response_to_latest:
            prompt += f" (Replying to [msgid:{len(history) - 1}]): "

        _logger.debug(f"Working prompt for channel #{channel.name}:\n{prompt}")

        if ctx.channel_config.typing_indicator:
            async with ctx.channel.typing():
                response, reply_to = await self.generate_response(prompt, ctx=ctx)
        else:
            response, reply_to = await self.generate_response(prompt, ctx=ctx)

        _logger.debug(f"Sending message in #{channel.name}:\n{response.content}")
        ctx.previously_sent_cache.append(response.content)

        allowed_mentions = discord.AllowedMentions(replied_user=True, users=True, everyone=False, roles=False)
        # TODO: Do something more intelligent than just chopping the string
        if reply_to is None:
            await channel.send(
                content=chop_string(response.content, 2000),
                allowed_mentions=allowed_mentions,
                files=response.attachments, embeds=response.embeds,
            )
        else:
            await reply_to.reply(
                content=chop_string(response.content, 2000),
                allowed_mentions=allowed_mentions,
                files=response.attachments, embeds=response.embeds,
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
            processed = await process_incoming(message, ctx=ctx)
            if processed is None:
                _logger.warning(f"Message #{message_index} ({message.id}) could not be processed. Skipping.")
                continue
            working_string += processed
            processed_history.append(working_string)
        return processed_history

    async def generate_response(
        self, 
        prompt: str, 
        *, 
        ctx: CheckupContext,
    ) -> tuple[DiscordResponse, discord.Message | None]:
        # We simply retry until we get a valid response from the LLM. It can be finicky sometimes.
        temperature: float = ctx.config.response_temperature
        for attempt_index in range(MAX_GENERATION_RETRIES):
            if attempt_index > 0:
                temperature = temperature + ctx.config.response_temperature_increment
                _logger.debug(f"Retrying LLM generation (attempt {attempt_index + 1}) with temperature {temperature:.2f}.")
            
            result = await ctx.apis.ollama.generate(
                model=ctx.config.models.text,
                prompt=prompt,
                options=ollama_api.Options(
                    stop=["\n[msgid:"],
                    num_predict=ctx.config.max_token_count,
                    temperature=temperature,
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
                reply_prefix, reply_suffix = "(replying to [msgid:", "]): "
                if not result.response.lstrip().startswith(reply_prefix):
                    content = result.response.strip().removeprefix(": ")
                else:
                    reply_i_str, _, content = result.response.lstrip().removeprefix(reply_prefix).partition(reply_suffix)
                    try:
                        replying_to_index = int(reply_i_str.strip())
                    except ValueError:
                        _logger.debug(f"Invalid reply index in response: {reply_i_str!r}. Skipping this response.")
                        continue
                    if replying_to_index < 0 or replying_to_index >= len(ctx.history):
                        _logger.debug(f"Reply index {replying_to_index} out of bounds for history length {len(ctx.history)}. Skipping this response.")
                        continue

            if content.strip() in ("", ":"):
                _logger.debug(f"Empty response from LLM on attempt {attempt_index + 1}. Retrying...")
                continue

            response: DiscordResponse | None = await process_outgoing(content, ctx=ctx)
            if response is None:
                _logger.debug(f"Processed content is None for response: {result.response!r}. Skipping this response.")
                continue
            return response, ctx.history[replying_to_index] if replying_to_index is not None else None

        raise RuntimeError(f"Failed to generate a valid response after {MAX_GENERATION_RETRIES} attempts.")


async def setup(bot: LLMBot) -> None:
    await bot.add_cog(AIBotFunctionality(bot))
