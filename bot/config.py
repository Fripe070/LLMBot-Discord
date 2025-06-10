import datetime
import json
import logging
import random
from pathlib import Path
from typing import Annotated

import pydantic

_logger = logging.getLogger(__name__)

class ConfigError(ValueError):
    """Custom exception for configuration-related errors."""
    def __init__(self, message: str):
        super().__init__(message)


def safe_write(file_path: Path, content: str) -> None:
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    backup_path = file_path.with_suffix(file_path.suffix + ".backup")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as tmp_file:
        tmp_file.write(content)

    if file_path.is_file():
        file_path.replace(backup_path)
    tmp_path.replace(file_path)

def valid_snowflake(value: int) -> int:
    """Validate that the value is a valid Discord snowflake."""
    if not (0 < value < 2**64):
        raise ValueError(f"Snowflake out of range 0 < {value} < 2^64")
    return value

class BaseConfigModel(pydantic.BaseModel):
    class Config:
        strict = True
        validate_default = True
        extra = "allow"
        frozen = True


IntTimeDelta = Annotated[
    datetime.timedelta,
    pydantic.PlainSerializer(lambda td: int(td.total_seconds()), when_used="json"),
    pydantic.BeforeValidator(lambda x: x if isinstance(x, datetime.timedelta) else datetime.timedelta(seconds=x)),
]
Snowflake = Annotated[int, pydantic.AfterValidator(valid_snowflake)]
DUMMY_SNOWFLAKE: Snowflake = 1


class ChannelConfig(BaseConfigModel):
    channel_id: Snowflake = DUMMY_SNOWFLAKE

    checkup_interval: IntTimeDelta = datetime.timedelta(minutes=5)
    checkup_variance: IntTimeDelta = datetime.timedelta(minutes=2)

    def random_interval(self) -> datetime.timedelta:
        variance: float = self.checkup_variance.total_seconds()
        return self.checkup_interval + datetime.timedelta(seconds=random.uniform(-variance, variance))

    talk_as_bot: bool = True # TODO: Rework this entire system. It is far too nuanced to be a single boolean.
    typing_indicator: bool = True

    history: pydantic.PositiveInt = 50
    history_when_responding: pydantic.PositiveInt = 20
    @property
    def max_history(self) -> int:
        """Maximum history to keep in memory."""
        return max(self.history, self.history_when_responding)

    class ActivityLimit(BaseConfigModel):
        window_size: int = 20
        max_bot_messages: int = 3
    activity_limit: ActivityLimit = ActivityLimit()


class BotConfig(BaseConfigModel):
    token: str = "<bot token goes here>"
    prefix: str | list[str] = "ai!"

    google_api_key: str = "<google api key goes here>"
    ollama_url: str = "http://localhost:11434"
    searx_url: str = "<searx instance URL goes here>"

    class ModelConfig(BaseConfigModel):
        text: str = "llama3.1:8b-text-q4_K_M"
        chat: str = "llama3.1:8b-instruct-q4_K_M"
    models: ModelConfig = ModelConfig()

    channels: Annotated[
        dict[Snowflake, ChannelConfig],
        pydantic.PlainSerializer(lambda chs: list(chs.values())),
        pydantic.BeforeValidator(lambda chs: {c["channel_id"]: c for c in chs} if isinstance(chs, list) else chs),
    ] = {DUMMY_SNOWFLAKE: ChannelConfig()}

    @classmethod
    def load(cls, config_path: Path | None = None) -> "BotConfig":
        """Load settings from a JSON file."""
        file_path = Path("config.json") if config_path is None else config_path
        default_config = cls()

        if not file_path.is_file():
            safe_write(file_path, default_config.model_dump_json(indent=4))
            raise FileNotFoundError(
                f"Configuration file not found at {file_path}. "
                f"A new file has been created with default settings."
            )

        with file_path.open("r", encoding="utf-8") as file:
            config_json: str = file.read()

        try:
            config_raw = json.loads(config_json)
        except json.JSONDecodeError as error:
            raise ConfigError(
                f"Configuration file at {file_path} is not valid JSON: {error}"
            ) from error
        try:
            config = cls.model_validate_json(config_json)
        except pydantic.ValidationError as error:
            raise ConfigError(
                f"Configuration file at {file_path} is invalid: {error}"
            ) from error

        # Check if the config changed compared to the raw json loaded
        config_dumped = config.model_dump(mode="json")
        if config_dumped != config_raw:
            safe_write(file_path, config.model_dump_json(indent=4))
            _logger.warning("An updated configuration file has been saved with default values for missing fields.")

        return config
