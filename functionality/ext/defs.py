import dataclasses
from collections.abc import Sequence

import discord
import ollama as ollama_api

from bot.config import ChannelConfig, BotConfig

MAX_GENERATION_RETRIES: int = 25

type DiscordSnowflake = int
type ChannelID = DiscordSnowflake
type MessageID = DiscordSnowflake

@dataclasses.dataclass(kw_only=True)
class APISet:
    ollama: ollama_api.AsyncClient

@dataclasses.dataclass(kw_only=True)
class CheckupContext:
    config: BotConfig
    channel_config: ChannelConfig
    apis: APISet
    channel: discord.TextChannel
    history: Sequence[discord.Message]
    author_indexes: dict[DiscordSnowflake, int] = dataclasses.field(default_factory=dict)
    is_response_to_latest: bool
    