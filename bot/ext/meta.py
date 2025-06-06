from discord.ext import commands

from bot.bot import LLMBot


class AIBotMeta(commands.Cog):    
    def __init__(self, bot: LLMBot) -> None:
        self.bot = bot
        
    @commands.command()
    @commands.is_owner()
    async def sync(self, ctx: commands.Context) -> None:
        """Sync the bot's application commands."""
        try:
            await self.bot.tree.sync()
            await ctx.send("Application commands synced successfully.")
        except Exception as e:
            await ctx.send(f"Failed to sync application commands: {e}")

async def setup(bot: LLMBot) -> None:
    await bot.add_cog(AIBotMeta(bot))
