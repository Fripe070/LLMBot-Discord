import collections
import dataclasses
from collections.abc import Sequence

import aiohttp
import discord
import ollama as ollama_api

from bot.config import ChannelConfig, BotConfig
from ..apis.image_gen import HordeSession

MAX_GENERATION_RETRIES: int = 25

type DiscordSnowflake = int
type ChannelID = DiscordSnowflake
type MessageID = DiscordSnowflake

@dataclasses.dataclass(kw_only=True)
class APISet:
    ollama: ollama_api.AsyncClient
    session: aiohttp.ClientSession
    horde: HordeSession

@dataclasses.dataclass(kw_only=True)
class CheckupContext:
    config: BotConfig
    channel_config: ChannelConfig
    apis: APISet
    channel: discord.TextChannel
    history: Sequence[discord.Message]
    author_indexes: dict[DiscordSnowflake, int] = dataclasses.field(default_factory=dict)
    previously_sent_cache: collections.deque[str]
    is_response_to_latest: bool

class NullAsyncContext:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
