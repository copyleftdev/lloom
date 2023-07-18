import pystache


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
