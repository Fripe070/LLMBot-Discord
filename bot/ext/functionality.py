import asyncio
import collections
import datetime
import random
import re
import time
from collections.abc import Sequence
from typing import TypedDict, cast

import discord
import ollama as ollama_api
from discord import app_commands
from discord.ext import commands
from yarl import URL

import bot.apis as apis
from bot.bot import LLMBot
from bot.util.formatting import chop_string, URL_REGEX, URL_LINK_REGEX, URL_REGEX_NO_EMBED

type DiscordSnowflake = int
type ChannelID = DiscordSnowflake
type MessageID = DiscordSnowflake

class ChannelSettings(TypedDict):
    interval: float
    jitter: float

def get_default_channel_settings() -> ChannelSettings:
    return ChannelSettings(
        interval=AIBotFunctionality.DEFAULT_INTERVAL.total_seconds(),
        jitter=AIBotFunctionality.DEFAULT_JITTER.total_seconds(),
    )

class AIBotFunctionality(commands.Cog):
    def __init__(self, bot: LLMBot) -> None:
        self.bot = bot
        self.bg_task: asyncio.Task | None = None
        self._cached_channel_settings: dict[ChannelID, ChannelSettings] | None = None

        self.schedule: dict[ChannelID, datetime.datetime] = {}

        self.max_message_history: int = bot.config.setdefault("message_history_max" , 100)
        self.max_message_history_reply: int = bot.config.setdefault("message_history_reply", 10)

        self.message_map: dict[ChannelID, collections.deque[discord.Message]] = collections.defaultdict(
            lambda: collections.deque(maxlen=max(self.max_message_history, self.max_message_history_reply))
        )

        self.bot.config.setdefault("allowed_channels", {})
        self.ollama = ollama_api.AsyncClient(
            host=bot.config.setdefault("ollama_base_url", "http://localhost:11434"),
        )
        self.searx_instance_url: str = bot.config.setdefault("searx_instance_url", "")
        self.tenor_api_key: str = bot.config.setdefault("tenor_api_key", "").strip()

        self.llm_model_text = bot.config.setdefault("llm_model_text", "llama3.1:8b-instruct-q4_K_M")
        self.llm_model_chat = bot.config.setdefault("llm_model_chat", "llama3.1:8b-text-q4_K_M")
        bot.config.setdefault("use_ai_user", True)
        bot.config.setdefault("activity_limits", {
            "window": 20,
            "limit": 3,
        })

    DEFAULT_INTERVAL = datetime.timedelta(minutes=10)
    DEFAULT_JITTER = datetime.timedelta(minutes=5)

    @property
    def channel_settings(self) -> dict[ChannelID, ChannelSettings]:
        if self._cached_channel_settings is not None:
            return self._cached_channel_settings

        channels_dict: dict[ChannelID, ChannelSettings] = {}
        self.bot.config.setdefault("allowed_channels", {})
        for channel_id, settings in self.bot.config["allowed_channels"].items():
            if not isinstance(settings, dict):
                raise TypeError(f"Expected checkup interval settings to be a dict, got {type(settings).__name__}")
            settings = cast(ChannelSettings, get_default_channel_settings() | settings)
            self.bot.config["allowed_channels"][str(channel_id)] = dict(settings)
            channels_dict[int(channel_id)] = settings

        self._cached_channel_settings = channels_dict
        return channels_dict

    async def cog_load(self) -> None:
        def runner_done_callback(task: asyncio.Task[None]) -> None:
            try:
                task.result()  # Raises if the task failed
            except asyncio.CancelledError:
                print("Checkup runner task was cancelled.")
            finally:
                self.bg_task = None

        self.bg_task = self.bot.loop.create_task(self.checkup_runner())
        self.bg_task.add_done_callback(runner_done_callback)

    async def cog_unload(self) -> None:
        if self.bg_task is not None and not self.bg_task.done():
            self.bg_task.cancel()
        self.bg_task = None

        self._cached_channel_settings = None

    @commands.Cog.listener("on_message")
    async def on_message(self, message: discord.Message) -> None:
        if message.channel.id not in self.channel_settings.keys():
            return
        if not isinstance(message.channel, discord.TextChannel):
            print(f"WARN: Received message in non-text channel {message.channel.id}, ignoring.")
            return

        # TODO: Temporary. Will exclude images which I dont rly want. Solution until embeds/attachments are handled
        if not message.content.strip():
            return

        self.message_map[message.channel.id].append(message)

        assert self.bot.user is not None, "Bot user must be set"
        if self.bot.user in message.mentions or (
            message.reference is not None and message.reference.cached_message is not None
            and message.reference.cached_message.author.id == self.bot.user.id
        ):
            await self.random_channel_checkup(
                message.channel,
                reply_to=message,  # Reply to the message that triggered the checkup
            )

    async def ensure_downloaded(self, model_name: str) -> None:
        def get_model_name(model: ollama_api.ListResponse.Model) -> str:
            if model.model is None:
                return ""
            name, _, version = model.model.partition(":")
            if version == "latest":
                return name
            return f"{name}:{version}"

        def models_match(a: ollama_api.ListResponse.Model | str, b: ollama_api.ListResponse.Model | str) -> bool:
            if isinstance(a, ollama_api.ListResponse.Model):
                a = get_model_name(a)
            if isinstance(b, ollama_api.ListResponse.Model):
                b = get_model_name(b)
            if not a or not b:
                return False
            return a == b

        if any(models_match(model, model_name) for model in (await self.ollama.list()).models):
            print(f'Skipping download of "{self.llm_model_text}" as it is already downloaded.')
        else:
            async for step in await self.ollama.pull(model_name, stream=True):
                print(f'Downloading model "{model_name}": {step.status}')

    async def checkup_runner(self) -> None:
        await self.ensure_downloaded(self.llm_model_text)
        await self.ensure_downloaded(self.llm_model_chat)

        await self.bot.wait_until_ready()

        while True:
            # TODO: Remove this print statement
            print("Currently scheduled:")
            for channel_id, next_time in self.schedule.items():
                print(f"- #{self.bot.get_channel(channel_id)} in {next_time - datetime.datetime.now()}")

            iter_start = time.time()
            for channel_id in self.channel_settings.keys():
                if channel_id in self.schedule:
                    next_time = self.schedule[channel_id]
                    if next_time > datetime.datetime.now():
                        continue
                else:
                    print(f"Channel {channel_id} not in schedule, scheduling now.")

                channel = self.bot.get_channel(channel_id) or await self.bot.fetch_channel(channel_id)
                assert isinstance(channel, discord.TextChannel), "Expected channel to be a TextChannel"
                await self.random_channel_checkup(channel)
                self.schedule[channel_id] = self.get_checkup_time(channel_id)
                print(f"Scheduled next checkup for {channel.name} at {self.schedule[channel_id]}")

            time_taken = time.time() - iter_start
            wait_time = max(0.0, 20.0 - time_taken)
            await asyncio.sleep(wait_time)

    def get_checkup_time(self, channel_id: int) -> datetime.datetime:
        next_time = datetime.datetime.now() + datetime.timedelta(seconds=self.channel_settings[channel_id]["interval"])
        next_time += datetime.timedelta(
            seconds=random.uniform(
                -self.channel_settings[channel_id]["jitter"],
                self.channel_settings[channel_id]["jitter"],
            )
        )
        return next_time

    async def random_channel_checkup(
        self,
        channel: discord.TextChannel,
        reply_to: discord.Message | None = None,
        *,
        history_override: Sequence[discord.Message] | None = None,
    ) -> None:
        assert channel.id in self.channel_settings, f"Channel {channel.id} is not an allowed channel"
        print(f"Performing checkup for {channel.name}...")

        relevant_history_length = self.max_message_history_reply if reply_to else self.max_message_history
        collected_messages: Sequence[discord.Message]
        if history_override is not None:
            collected_messages = history_override[-relevant_history_length:]
        else:
            collected_messages = tuple(self.message_map[channel.id])[-relevant_history_length:]
            if not collected_messages:
                print(f"No messages collected in {channel.name}. Trying to fetch {self.max_message_history} messages.")
                collected_messages = [
                    msg
                    async for msg in channel.history(
                        limit=relevant_history_length,
                        before=reply_to or datetime.datetime.now(),
                    )
                ][::-1]  # Reverse to have oldest first

        msg_process_start = time.perf_counter()

        assert self.bot.user is not None, "Bot user must be set"
        if reply_to is None and self.bot.config["activity_limits"]["limit"] > 0:
            bot_sighting_count: int = 0
            for i in range(0, -min(self.bot.config["activity_limits"]["window"], len(collected_messages)), -1):
                if collected_messages[i].author.id == self.bot.user.id:
                    bot_sighting_count += 1
                    if bot_sighting_count >= self.bot.config["activity_limits"]["limit"]:
                        print(f"Skipping checkup for {channel.name} as the bot is too active.")
                        return

        author_indexes: dict[DiscordSnowflake, int] = {}
        handled_msg_ids: list[MessageID] = []
        handled_rows: list[str] = []

        for msg_index, message in enumerate(collected_messages):
            if reply_to is not None and message.id > reply_to.id: # Larger IDs are newer
                break

            author_index = author_indexes.setdefault(message.webhook_id or message.author.id, len(author_indexes))
            history_string = f"[msgid:{msg_index}] User {author_index}"

            if message.reference is not None:
                referenced_index: int | None = None
                for i, handled_msg_id in enumerate(handled_msg_ids):
                    if handled_msg_id == message.reference.message_id:
                        referenced_index = i
                        break
                if referenced_index is not None:
                    assert collected_messages[referenced_index].id == message.reference.message_id, (
                        f"Collected message ID does not match reference: "
                        f"{collected_messages[referenced_index].id} != {message.reference.message_id}"
                    )
                    history_string += f" (replying to [msgid:{referenced_index}])"

            history_string += ": "
            history_string += await self.process_incoming(message, author_indexes=author_indexes)

            handled_rows.append(history_string)
            handled_msg_ids.append(message.id)

        prompt: str = (
            "The following is the chat history of a Discord channel. Markdown is supported.\n"
            "Mention format: `@UserID`. For example: @1 for User 1.\n"
        )
        if channel.guild.emojis:
            emojis = " ".join(f":{emoji.name}:" for emoji in channel.guild.emojis)
            prompt += f"Chatroom emotes: {emojis}\n"

        prompt += "\n" + "\n".join(handled_rows) + "\n"

        # Enabling the config option would result in the bot acting more consistently with itself
        ai_user_id: int | None = None
        if self.bot.config["use_ai_user"] and self.bot.user:
            ai_user_id = author_indexes.setdefault(self.bot.user.id, len(author_indexes))

        prompt += f"[msgid:{len(handled_msg_ids)}] User {"" if ai_user_id is None else ai_user_id}"
        if reply_to is not None:
            prompt += f" (Replying to [msgid:{len(handled_msg_ids) - 1}])"

        msg_process_end = time.perf_counter()
        print(f"Processed {len(collected_messages)} messages in {msg_process_end - msg_process_start:.2f} seconds.")

        print(prompt)
        print(repr(prompt))

        # We simply retry until we get a valid response from the LLM.
        response_content: str | None = None
        replied_msg_index: int | None = None
        while response_content is None:
            replied_msg_index = None

            result = await self.ollama.generate(
                model=self.llm_model_text,
                prompt=prompt,
                options=ollama_api.Options(
                    stop=["\n[msgid:"],
                    num_predict=500,
                ),
            )

            meta, _, content = result.response.partition(": ")
            content = content.strip()
            if not content:
                continue
            # THe first part should be the user ID, but we don't use it so it's ignored
            predicted_user_id, _, reply_str = meta.partition(" ")
            if self.bot.config["use_ai_user"] and predicted_user_id:
                continue # If we already have a predetermined user ID, the bot is not allowed to append to it

            if reply_str:
                if not reply_str.startswith("Replying to [msgid:") or not reply_str.endswith("])"):
                    continue  # Invalid response format
                try:
                    replied_msg_index = int(reply_str[len("Replying to [msgid:") : -len("])")])
                except ValueError:
                    continue
                if replied_msg_index < 0 or replied_msg_index >= len(handled_msg_ids):
                    continue
            if reply_to is not None:
                replied_msg_index = len(handled_msg_ids) - 1

            print("Unprocessed LLM response:", content)

            response_content = await self.process_outgoing(
                content,
                channel=channel,
                author_indexes=author_indexes,
                history_context=collected_messages,
            )

        print(f"Final LLM response: {response_content}")

        allowed_mentions = discord.AllowedMentions(replied_user=True, users=True, everyone=False, roles=False)
        if replied_msg_index is None:
            await channel.send(
                content=chop_string(response_content, 2000),
                allowed_mentions=allowed_mentions,
            )
        else:
            await collected_messages[replied_msg_index].reply(
                content=chop_string(response_content, 2000),
                allowed_mentions=allowed_mentions,
            )

    async def process_incoming(
        self,
        message: discord.Message,
        *,
        author_indexes: dict[DiscordSnowflake, int],
    ) -> str:
        # TODO: Surface more message data (e.g. attachments, embeds, images)
        content = message.content.strip()
        assert message.guild is not None, "Message must be in a guild"

        emoji_map = {str(emoji): f":{emoji.name}:" for emoji in message.guild.emojis}
        for emoji_str, emoji_rep in emoji_map.items():
            content = content.replace(emoji_str, emoji_rep)

        # Replace mentions with their index in the author_indexes
        for match_full, match_snowflake in re.findall(r"(<@!?([0-9]+)>)", content):
            mapped_snowflake = author_indexes.setdefault(int(match_snowflake), len(author_indexes))
            content = content.replace(match_full, f"@{mapped_snowflake}")

        return content

    async def process_outgoing(
        self,
        content: str,
        *,
        channel: discord.TextChannel,
        author_indexes: dict[DiscordSnowflake, int],
        history_context: Sequence[discord.Message],
    ) -> str | None:
        # TODO: *Technically* we should have a check for if the content is in a codeblock

        # Almost guaranteed to be an invalid response
        if re.search(r"[msgid:[0-9]+]", content):
            return None

        # Replace emotes with their Discord representation
        emoji_map = {f":{emoji.name}:": emoji for emoji in channel.guild.emojis}
        for emoji_str, emoji in emoji_map.items():
            content = content.replace(emoji_str, str(emoji))

        inverse_author_indexes = {v: k for k, v in author_indexes.items()}
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
            *(self.process_url(url, history_context=history_context) for url in urls),
            return_exceptions=True,
        )
        for original_url, processed_url in zip(urls, processed_urls):
            if processed_url is None:
                continue
            if not isinstance(processed_url, str):
                print(f"Error processing URL {original_url}: {processed_url}")
                continue
            print(f"{original_url} -> {processed_url}")
            content = content.replace(original_url, processed_url)

        # TODO: Images


        return content

    async def process_url(
        self,
        url: str,
        *,
        history_context: Sequence[discord.Message],
    ) -> str | None:
        try:
            parsed_url: URL = URL(url)
        except ValueError:
            return None
        if parsed_url.scheme not in {"http", "https"}:
            return None
        if not parsed_url.host:
            return None

        if self.tenor_api_key and (
            parsed_url.host.endswith(("tenor.com", "giphy.com")) 
            or parsed_url.suffix in {".gif", ".webp"}
        ):
            tenor_query = await apis.searching.generate_search_query(
                "\n".join(msg.content for msg in history_context),
                prompt=apis.searching.GIF_PROMPT,
                model=self.bot.config["llm_model_chat"],
                ollama_api=self.ollama,
            )
            print(f"Searching Tenor for: {tenor_query}")
            search_results = await apis.searching.search_tenor(
                tenor_query,
                api_key=self.tenor_api_key,
                limit=3,
            )
            if not search_results:
                raise ValueError(f"No Tenor results found for query: {tenor_query}")
            return str(random.choice(search_results).url)

        # Any engines that use normal search queries are handled below
        if not self.searx_instance_url:
            print(f"Skipping URL {url} as no Searx instance URL is configured.")
            return url
        query = await apis.searching.generate_search_query(
            "\n".join(msg.content for msg in history_context),
            model=self.bot.config["llm_model_chat"],
            ollama_api=self.ollama,
        )
        print(f"Generated query for URL {url}: {query}")
        
        if parsed_url.host.endswith(("youtube.com", "youtu.be")):
            search_results = await apis.searching.search_searx(
                query,
                api_url=self.searx_instance_url,
                engines=["youtube"]
            )
            if not search_results:
                raise ValueError(f"No YouTube results found for query: {query}")
            return str(search_results[0].url)

        # We just search for the query in general search engines
        search_results = await apis.searching.search_searx(query, api_url=self.searx_instance_url)
        if not search_results:
            raise ValueError(f"No search results found for query: {query}")
        return str(search_results[0].url)

    cmd_group = app_commands.Group(
        name="channel",
        description="Manage AI bot channels",
        guild_only=True,
        default_permissions=discord.Permissions(manage_channels=True),
    )

    @cmd_group.command(name="add", description="Add a channel to the AI bot's list of allowed channels")
    async def add_channel(self, interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        if channel.id in self.channel_settings:
            await interaction.response.send_message(f"{channel.mention} is already an allowed channel.", ephemeral=True)
            return
        self.bot.config["allowed_channels"][str(channel.id)] = get_default_channel_settings()
        self._cached_channel_settings = None  # Invalidate cache

        await interaction.response.send_message(f"Added {channel.mention} to allowed channels.", ephemeral=True)

    @cmd_group.command(name="remove", description="Remove a channel from the AI bot's list of allowed channels")
    async def remove_channel(self, interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        if channel.id not in self.channel_settings:
            await interaction.response.send_message(f"{channel.mention} is not an allowed channel.", ephemeral=True)
            return

        del self.bot.config["allowed_channels"][str(channel.id)]
        self._cached_channel_settings = None  # Invalidate cache
        if channel.id in self.schedule:
            del self.schedule[channel.id]
        if channel.id in self.message_map:
            del self.message_map[channel.id]

        await interaction.response.send_message(f"Removed {channel.mention} from allowed channels.", ephemeral=True)

    @cmd_group.command(name="interval", description="Set the checkup interval for a channel")
    async def set_interval(self, interaction: discord.Interaction, channel: discord.TextChannel, minutes: int) -> None:
        if channel.id not in self.channel_settings:
            await interaction.response.send_message(f"{channel.mention} is not an allowed channel.", ephemeral=True)
            return
        if minutes < 0:
            await interaction.response.send_message("Interval must be non-negative.", ephemeral=True)
            return

        # Settings will by now be populated by the earlier settings property access, so we can assume the key exists
        self.channel_settings[channel.id]["interval"] = minutes * 60.0
        self._cached_channel_settings = None  # Invalidate cache

        await interaction.response.send_message(
            f"Set checkup interval for {channel.mention} to {minutes} minutes.", ephemeral=True
        )

    @cmd_group.command(name="jitter", description="Set the maximum interval variation (jitter) for a channel")
    async def set_jitter(self, interaction: discord.Interaction, channel: discord.TextChannel, minutes: float) -> None:
        if channel.id not in self.channel_settings:
            await interaction.response.send_message(f"{channel.mention} is not an allowed channel.", ephemeral=True)
            return
        minutes = abs(minutes)

        # Settings will by now be populated by the earlier settings property access, so we can assume the key exists
        self.channel_settings[channel.id]["jitter"] = minutes * 60.0
        self._cached_channel_settings = None

        await interaction.response.send_message(f"Set jitter for {channel.mention} to {minutes} minutes.", ephemeral=True)


async def setup(bot: LLMBot) -> None:
    await bot.add_cog(AIBotFunctionality(bot))
