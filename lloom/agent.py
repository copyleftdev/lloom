import os

import openai
import pystache


class Agent:
    def __init__(
        self,
        name: str,
        system_statement: str,
        prompt: str,
        input: list[str],
        model: None,
    ):
        self.name = name
        self.model_type = name.replace("openai/", "")
        self.system_statement = system_statement
        self.prompt = prompt
        self.input = input
        if model is None:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            self.model = openai.ChatCompletion
        else:
            self.model = model

    def prepare_prompt(self, input_dict: dict[str, str]) -> str:
        if set(self.input) != set(input_dict.keys()):
            raise ValueError("Keys in input dictionary do not match expected keys.")

        return pystache.render(self.prompt, input_dict)

    def generate_response(self, prompt: str) -> str:
        output = self.model.create(
            model=self.model_type,
            messages=[
                {"role": "system", "content": self.system_statement},
                {"role": "user", "content": prompt},
            ],
        )
        return output["choices"][0]["message"]["content"]
