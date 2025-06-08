import json
import asyncio
import logging
from os import PathLike
from pathlib import Path

import discord
from discord.ext import commands

from .config import BotConfig

CONFIG_PATH: Path = Path("config.json").resolve()

class LLMBot(commands.Bot):
    def __init__(self) -> None:
        super().__init__(
            command_prefix=["ai!"], 
            intents=discord.Intents.all(),
        )
        self.config: BotConfig = BotConfig()  # Default config, will be populated upon bot startup

    async def setup_hook(self) -> None:
        await self.load_extension("functionality.ext") # Core functionality

    def run_llmbot(
        self,
        config_path: PathLike[str] | str = "config.json",
        log_directory: PathLike[str] | str = "logs/",
    ) -> None:
        self.config = BotConfig.load(Path(config_path))

    async def close(self) -> None:
        self.save_config()
        
        await super().close()

        async def runner():
            async with self:
                await self.start(self.config.token, reconnect=True)

        try:
            asyncio.run(runner())
        except KeyboardInterrupt:
            return
