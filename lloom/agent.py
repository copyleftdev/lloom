import pystache
import requests


class Agent:
    def __init__(
        self,
        name: str,
        model: any,
        system_statement: str,
        prompt: str,
        input: list[str],
    ):
        self.name = name
        self.model = model
        self.system_statement = system_statement
        self.prompt = prompt
        self.input = input

    def prepare_prompt(self, input_dict: dict[str, str]) -> str:
        if set(self.input) != set(input_dict.keys()):
            raise ValueError("Keys in input dictionary do not match expected keys.")

        return pystache.render(self.prompt, input_dict)

    def generate_response(self, prompt: str) -> str:
        return self.model.generate(prompt=prompt)

    def generate(self, *args: any, **kwargs: any) -> any:
        self._prepare_headers()
        input_data = self._prepare_input(*args, **kwargs)
        raw_output = self._send_request(input_data)
        return self._parse_output(raw_output)

    def _send_request(self, input_data: dict[str, any]) -> dict[str, any]:
        retries = 0
        while retries <= self.max_retries:
            try:
                response = requests.post(
                    self.url, headers=self.headers, json=input_data
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException:
                sleep(2**retries)
                retries += 1
        raise Exception(f"Max retries exceeded for POST request to {self.url}")


class ChatModel(Agent):
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

    def _prepare_headers(self) -> None:
        self.headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        }
        if self.organization:
            self.headers["OpenAI-Organization"] = self.organization

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
