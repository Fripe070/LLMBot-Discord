import datetime
import itertools
import logging
import logging.handlers
from pathlib import Path

import discord.utils


class LLMBotFormatter(logging.Formatter):
    ANSI_RESET: str = "\x1b[0m"         # RESET

    LEVEL_ANSI_MAP: dict[int, str] = {
        logging.DEBUG:    "\x1b[30;1m", # BLACK   FG  BOLD
        logging.INFO:     "\x1b[34;1m", # BLUE    FG  BOLD
        logging.WARNING:  "\x1b[33;1m", # YELLOW  FG  BOLD
        logging.ERROR:    "\x1b[31m",   # RED     FG
        logging.CRITICAL: "\x1b[41m",   # RED     BG
    }
    TIME_ANSI: str = "\x1b[30;1m"       # BLACK BG BOLD
    NAME_ANSI: str = "\x1b[35m"         # MAGENTA FG
    EXCEPTION_ANSI: str = "\x1b[31m"    # RED FG

    def __init__(self, use_colour: bool) -> None:
        super().__init__()
        self.use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        level_ansi = self.LEVEL_ANSI_MAP.get(record.levelno, self.ANSI_RESET)
        record.asctime = self.formatTime(record, "%Y-%m-%d %H:%M:%S")

        prefix_colourless = f"{record.asctime} {record.levelname:<8} {record.name}: "
        prefix_colourful = (
            f"{self.TIME_ANSI}{record.asctime     }{self.ANSI_RESET} "
            f"{level_ansi    }{record.levelname:<8}{self.ANSI_RESET} "  # "CRITICAL" is 8 chars long so we pad for it
            f"{self.NAME_ANSI}{record.name        }{self.ANSI_RESET}: "
        )

        message = record.getMessage().rstrip()
        if record.exc_info:
            if self.use_colour:
                message += "\n" + self.EXCEPTION_ANSI + self.formatException(record.exc_info).rstrip() + self.ANSI_RESET
            else:
                message += "\n" + self.formatException(record.exc_info).rstrip()
        if record.stack_info:
            message += "\n" + self.formatStack(record.stack_info).rstrip()
            
        message = self.apply_indent(message, 4)
        if self.use_colour:
            return prefix_colourful + message
        else:
            return prefix_colourless + message

    @staticmethod
    def apply_indent(text: str, depth: int) -> str:
        indent: str = " " * depth
        first_line, *other_lines = text.splitlines(keepends=True)
        return first_line + "".join(indent + line for line in other_lines)

def get_incremental_log_name(log_file: Path) -> Path:
    for i in itertools.count(start=1):
        rename_path = log_file.with_name(log_file.stem + f".{i:02d}" + log_file.suffix)
        if not rename_path.exists():
            return rename_path
    raise RuntimeError(f"Unable to find a unique name for log file {log_file}. This should never happen.")

def file_log_handler(logs_dir: Path) -> logging.FileHandler:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "latest.log"

    if log_file.is_file():
        old_log_timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d")
        timestamp_length: int = len(old_log_timestamp)
        with log_file.open(encoding="utf-8") as file:
            old_log_timestamp: str = file.read(timestamp_length).strip() or old_log_timestamp
        log_file.rename(get_incremental_log_name(log_file.with_stem(old_log_timestamp)))

    return logging.FileHandler(
        filename=log_file,
        mode="w",
        encoding="utf-8",
    )

def setup_logging(logs_dir: Path, *, level: int = logging.INFO) -> None:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LLMBotFormatter(
        use_colour=discord.utils.stream_supports_colour(console_handler.stream)
    ))
    file_handler = file_log_handler(logs_dir)
    file_handler.setFormatter(LLMBotFormatter(use_colour=False))

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=(console_handler, file_handler),
        force=True,  # Override any previous logging configuration
    )
