import json
from pathlib import Path

import discord
from discord.ext import commands


CONFIG_PATH: Path = Path("config.json").resolve()

class LLMBot(commands.Bot):
    def __init__(self) -> None:
        super().__init__(
            command_prefix=["ai!"], 
            intents=discord.Intents.all(),
        )
        self.config: dict = {}

    async def setup_hook(self) -> None:
        self.load_config()
        
        # Load the cogs containing most bot functionality
        await self.load_extension("bot.ext.meta")
        await self.load_extension("bot.ext.functionality")

    async def close(self) -> None:
        self.save_config()
        
        await super().close()

    def load_config(self) -> None:
        """Load the bot's configuration from the config.json file."""
        if CONFIG_PATH.is_file():
            try:
                with CONFIG_PATH.open("r") as config_file:
                    self.config = json.load(config_file)
            except json.JSONDecodeError:
                print(f"Error reading {CONFIG_PATH}.")

        if not self.config:
            print("Using default configuration.")
            self.config = {}
            
    def save_config(self) -> None:
        """Save the bot's configuration to the config.json file."""
        tmp_file = CONFIG_PATH.with_name(f"{CONFIG_PATH.stem}_tmp.{CONFIG_PATH.suffix}")
        with tmp_file.open("w") as config_file:
            json.dump(self.config, config_file, indent=4)
        if CONFIG_PATH.is_file():
            CONFIG_PATH.unlink()
        tmp_file.rename(CONFIG_PATH)
        print(f"Configuration saved to {CONFIG_PATH}.")
