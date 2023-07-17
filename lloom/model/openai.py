import os

from .abc import AbstractModel


class OpenAIModel(AbstractModel):
    organization: str = None

    def __init__(
        self,
        name: str,
        type: str,
        provider: str,
        url: str,
        organization: str,
        max_retries: int = 3,
    ):
        super().__init__(name, type, provider, url, max_retries)
        self.organization = organization

    def _prepare_headers(self) -> None:
        self.headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        }
        if self.organization:
            self.headers["OpenAI-Organization"] = self.organization


class AdaModel(OpenAIModel):
    def __init__(
        self,
        organization: str = None,
        max_retries: int = 3,
    ):
        self.name = "text-embedding-ada-002"
        self.kind = "embeddings"
        self.provider = "openai"
        self.url = "https://api.openai.com/v1/embeddings"
        self.organization = organization
        self.max_retries = max_retries

    def _prepare_input(self, text: str, user: str = None) -> dict[str, any]:
        input_data = {
            "model": self.name,
            "input": text,
        }
        if user:
            input_data["user"] = user
        return input_data

    def _parse_output(self, raw_output: dict[str, any]) -> any:
        if (
            "data" in raw_output
            and len(raw_output["data"]) > 0
            and "embedding" in raw_output["data"][0]
        ):
            return raw_output["data"][0]["embedding"]
        else:
            raise ValueError("Invalid response format.")


class ChatModel(OpenAIModel):
    def __init__(
        self,
        name: str,
        organization: str = None,
        suffix: str = None,
        max_tokens: int = 16,
        temperature: float = 1,
        top_p: float = 1,
        n: int = 1,
        stream: bool = False,
        logprobs: int = None,
        echo: bool = False,
        stop: str or list = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        best_of: int = 1,
        logit_bias: dict = None,
        max_retries: int = 3,
    ):
        self.name = name
        self.url = "https://api.openai.com/v1/completions"
        self.provider = "openai"
        self.kind = "chat"
        self.organization = organization

        self.suffix = suffix
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stream = stream
        self.logprobs = logprobs
        self.echo = echo
        self.stop = stop
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.best_of = best_of
        self.logit_bias = logit_bias
        self.max_retries = max_retries

    def _prepare_input(self, prompt: str, user: str = None) -> dict[str, any]:
        input_data = {
            "model": self.name,
            "prompt": prompt,
            "suffix": self.suffix,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stream": self.stream,
            "logprobs": self.logprobs,
            "echo": self.echo,
            "stop": self.stop,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "best_of": self.best_of,
            "logit_bias": self.logit_bias,
            "user": user,
        }
        return input_data

    def _parse_output(self, raw_output: dict[str, any]) -> any:
        if (
            "choices" in raw_output
            and len(raw_output["choices"]) > 0
            and "message" in raw_output["choices"][0]
            and "content" in raw_output["choices"][0]["message"]
        ):
            return raw_output["choices"][0]["message"]["content"]
        else:
            raise ValueError("Invalid response format.")
