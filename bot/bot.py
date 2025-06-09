import asyncio
import logging
import sys
from os import PathLike
from pathlib import Path
from types import TracebackType

import discord
from discord.ext import commands

import functionality
from . import log_utils
from .config import BotConfig

_logger = logging.getLogger(__name__)

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
        log_utils.setup_logging(Path(log_directory), level=logging.INFO)
        logging.getLogger(functionality.__name__).setLevel(logging.DEBUG)

        # ty andrew xoxo
        def handle_exception(exc_type: type[BaseException], value: BaseException, traceback: TracebackType) -> None:
            _logger.critical(f"Uncaught {exc_type.__name__}: {value}", exc_info=(exc_type, value, traceback))
        sys.excepthook = handle_exception
        
        self.config = BotConfig.load(Path(config_path))

        async def runner():
            async with self:
                await self.start(self.config.token, reconnect=True)

        try:
            asyncio.run(runner())
        except KeyboardInterrupt:
            return
