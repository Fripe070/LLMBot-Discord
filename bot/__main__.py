import os

from .bot import LLMBot

if __name__ == "__main__":
    bot = LLMBot()
    bot.run(token=os.environ.get("DISCORD_TOKEN", ""))