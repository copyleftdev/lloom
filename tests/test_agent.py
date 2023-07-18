import pytest

from lloom.agent import Agent


class SimpleChatModel:
    def generate(self, prompt):
        return prompt


@pytest.fixture
def agent():
    model = SimpleChatModel()
    name = "helpful_assistant"
    system_statement = "I am an AI with a specialty in American political science. How can I assist you today?"  # noqa: E501
    prompt = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{{context}}\n\nQuestion: {{query}}\nHelpful Answer:"  # noqa: E501
    input_keys = ["context", "query"]
    return Agent(name, model, system_statement, prompt, input_keys)


def test_agent_prepare_prompt(agent):
    input_dict = {"context": "Test context", "query": "Test query"}
    expected_prompt = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nTest context\n\nQuestion: Test query\nHelpful Answer:"  # noqa: E501
    assert agent.prepare_prompt(input_dict) == expected_prompt

    input_dict = {"context": "Test context"}
    with pytest.raises(ValueError):
        agent.prepare_prompt(input_dict)


def test_agent_generate_response(agent):
    input_dict = {"context": "Test context", "query": "Test query"}
    prompt = agent.prepare_prompt(input_dict)
    expected_response = prompt
    assert agent.generate_response(prompt) == expected_response
