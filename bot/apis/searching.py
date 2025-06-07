import json
from dataclasses import dataclass
from enum import Enum

import ollama
from pydantic import BaseModel
from yarl import URL
import aiohttp

GENERIC_PROMPT = (
    "Based on the above message history, I want to share a relevant link. "
    "Please suggest search queries that would yield good and relevant results.\n"
    "Queries should focus on a single topic and include relevant keywords "
    "and enough context to get a good search result.\n"
    
    "A reasoning field is required to explain your thought process.\n"
    "After queries are generated, select the most fitting query and return it in the `most_fitting` field.\n"
)

GIF_PROMPT = (
    "Based on the above message history, I want to share a GIF. "
    "Please suggest search queries that would yield good and relevant results.\n"
    "Queries should focus on a single topic based on the conversation and should include relevant keywords "
    "along with enough context to get a good search result.\n"
    
    "A reasoning field is required to explain your thought process.\n"
    "After queries are generated, select the most fitting query and return it in the `most_fitting` field.\n"
)

class ResponseFormat(BaseModel):
    reasoning: str
    queries: list[str]
    most_fitting: str

async def generate_search_query(
    ctx: str,
    *,
    model: str,
    ollama_api: ollama.AsyncClient = ollama.AsyncClient(),
    prompt: str = GENERIC_PROMPT,
    max_retries: int = 3,
) -> str:
    for _ in range(max_retries):
        resp = await ollama_api.chat(
            model=model,
            messages=[ollama.Message(role="user", content=f"{ctx}\n\n" + prompt)],
            format=ResponseFormat.model_json_schema(),
        )
        try:
            return json.loads(resp.message.content or "{}")["most_fitting"].strip()
        except json.JSONDecodeError:
            continue
    raise ValueError("Failed to generate a valid search query after multiple attempts.")
    

class SearchCategory(Enum):
    GENERAL = "general"
    IMAGES = "images"
    VIDEOS = "videos"
    NEWS = "news"
    MAP = "map"
    MUSIC = "music"
    IT = "it"
    SCIENCE = "science"
    FILES = "files"
    SOCIAL_MEDIA = "social media"

class SafeSearchLevel(Enum):
    NONE = "0"
    MODERATE = "1"
    STRICT = "2"


@dataclass
class SearchResult:
    url: URL
    page_title: str
    engine: str

    def __str__(self) -> str:
        return f"{self.page_title} ({self.engine}): {self.url}"


async def search_searx(
    query: str,
    *,
    api_url: URL | str,
    engines: list[SearchCategory | str] | None = None,
    safe_search: SafeSearchLevel = SafeSearchLevel.MODERATE,
) -> list[SearchResult]:
    engine_bangs: list[str] = [
        "!" + (cat.value if isinstance(cat, SearchCategory) else cat)
        for cat in (engines or [])
    ]

    async with aiohttp.ClientSession() as session:
        async with session.get(
            URL(api_url) / "search",
            params={
                "q": f"{" ".join(engine_bangs)} {query}",
                "format": "json",
                "safesearch": safe_search.value,
            },
        ) as response:
            response.raise_for_status()
            json_resp = await response.json()
            return [
                SearchResult(
                    url=URL(result["url"]),
                    page_title=result["title"],
                    engine=result["engine"],
                )
                for result in json_resp.get("results", [])
            ]

async def search_tenor(
    query: str,
    *,
    api_url: URL | str = "https://tenor.googleapis.com/v2/search",
    api_key: str,
    limit: int = 10,
    random: bool = False,
) -> list[SearchResult]:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            URL(api_url),
            params={
                "q": query,
                "key": api_key,
                "limit": limit,
                "media_filter": "minimal",
                "random": "true" if random else "false",
            },
        ) as response:
            response.raise_for_status()
            json_resp = await response.json()
            return [
                SearchResult(
                    url=URL(result["url"]),
                    page_title=result["content_description"],
                    engine="Tenor",
                )
                for result in json_resp.get("results", [])
            ]
