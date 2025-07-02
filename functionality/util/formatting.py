import datetime
import enum
import re

import discord


def link(url: str, cover_text: str, should_embed: bool = True, tooltip: str | None = None) -> str:
    """
    Format a link with optional embedding and tooltip.
    :param url: The URL to link to.
    :param cover_text: The text to display in place of the raw URL. Must not be empty.
    :param should_embed: Whether the link embed is hidden or not.
    :param tooltip: Optional tooltip text that appears on hover.
    :return: A formatted Markdown link string.
    """
    if not cover_text:
        raise ValueError("Cover text must not be empty.")
    url = url.strip().replace(")", "\\)")
    tooltip = f' "{tooltip}"' if tooltip else ""
    if should_embed:
        return f"[{cover_text}]({url}{tooltip})"
    else:
        return f"[{cover_text}](<{url}>{tooltip})"


class TimestampStyle(enum.Enum):
    """Enum for Discord timestamp styles."""
    t = "t"  # Short time
    T = "T"  # Long time
    d = "d"  # Short date
    D = "D"  # Long date
    f = "f"  # Short date and time
    F = "F"  # Long date and time
    R = "R"  # Relative time (e.g., "in 5 minutes")


def to_timestamp(dt: datetime.datetime, style: TimestampStyle = TimestampStyle.f) -> str:
    """
    Format a datetime object as a Discord timestamp.
    
    :param dt: The datetime object to format.
    :param style: The style of the timestamp.
    :return: A formatted string containing the Discord timestamp.
    """
    return f"<t:{int(dt.timestamp())}:{style.value}>"


def from_timestamp(timestamp: str) -> datetime.datetime:
    """
    Convert a Discord timestamp string back to a datetime object.
    
    :param timestamp: The Discord timestamp string (e.g., "<t:1234567890:R>").
    :return: A datetime object representing the timestamp.
    """
    match = re.match(r"<t:(\d+)(?::[tTdDfFR])?>", timestamp)
    if not match:
        raise ValueError("Invalid Discord timestamp format.")
    return datetime.datetime.fromtimestamp(int(match.group(1)), tz=datetime.timezone.utc)


# noinspection PyShadowingBuiltins
def chop_string(text: str, max_length: int, ellipsis: str = "\N{HORIZONTAL ELLIPSIS}") -> str:
    """
    Chop a string to a maximum length and append an ellipsis if it exceeds that length.
    
    :param text: The string to be chopped.
    :param max_length: The maximum length of the string before it is chopped.
    :param ellipsis: The string to append if the text is chopped (default is horizontal ellipsis).
    :return: The chopped string, possibly with an ellipsis appended.
    """
    if len(text) > max_length:
        return text[:max_length - len(ellipsis)] + ellipsis
    return text


# Pulled from discord's sources and modified slightly with capturing groups
URL_REGEX: re.Pattern = re.compile(r"(?P<url>https?:\/\/[^\s<]+[^<.,:;\"')\]\s])")
URL_REGEX_NO_EMBED: re.Pattern = re.compile(r"<(?P<url>[^:\ >]+:\/[^\ >]+)>")
URL_LINK_REGEX: re.Pattern = re.compile(
    r"""
    \[
    (?P<text>(?:\[[^\]]*\]|[^\[\]]|\](?=[^\[]*\]))*)
    \]\(\s*
    <?(?P<url>(?:\([^)]*\)|[^\s\\]|\\.)*?)>?
    (?P<tooltip>\s+['"](?:[\s\S]*?)['"])?
    \s*\)
    """,
    flags=re.VERBOSE,
)

def col_to_hex(colour: discord.Colour) -> str:
    return f"#{colour.r:02x}{colour.g:02x}{colour.b:02x}"
