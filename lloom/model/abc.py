from abc import ABC, abstractmethod
from time import sleep

import requests


class AbstractModel(ABC):
    def __init__(
        self, name: str, kind: str, provider: str, url: str, max_retries: int = 3
    ) -> None:
        self.name = name
        self.kind = kind
        self.provider = provider
        self.url = url
        self.max_retries = max_retries

    def generate(self, *args: any, **kwargs: any) -> any:
        self._prepare_headers()
        input_data = self._prepare_input(*args, **kwargs)
        raw_output = self._send_request(input_data)
        return self._parse_output(raw_output)

    @abstractmethod
    def _prepare_headers(self) -> None:
        pass

    @abstractmethod
    def _prepare_input(self, *args: any, **kwargs: any) -> dict[str, any]:
        pass

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

    @abstractmethod
    def _parse_output(self, raw_output: dict[str, any]) -> any:
        pass
