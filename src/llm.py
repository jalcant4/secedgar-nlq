import tomllib
from pathlib import Path

import requests

config_path = Path(__file__).with_name("config.toml")

try:
    with config_path.open("rb") as f:
        config = tomllib.load(f)
except OSError:
    raise SystemExit(
        f"{config_path} not found. Copy src/config.toml.example to "
        "src/config.toml and set [llm].endpoint."
    ) from None

ENDPOINT = config["llm"]["endpoint"]


def llm_query(prompt: str) -> str:
    models = requests.get(
        f"{ENDPOINT}/models",
        timeout=30
    )
    models.raise_for_status()
    model = models.json()["data"][0]["id"]

    response = requests.post(
        f"{ENDPOINT}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2048,
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]