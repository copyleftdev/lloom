from lloom import Lloom
from lloom.model.openai import ChatModel
from lloom.store.chroma import ChromaClient


def test_create_objects_from_yaml():
    file_path = "./tests/sotu.yml"
    lloom = Lloom(file_path=file_path)
    lloom.create_objects_from_yaml()

    assert len(lloom.models) == 1
    assert "turbo" in lloom.models

    assert len(lloom.stores) == 1
    assert "sotu_db" in lloom.stores

    assert len(lloom.datasets) == 1
    assert "sotu_raw" in lloom.datasets

    turbo_model = lloom.models["turbo"]
    assert turbo_model.name == "gpt-3.5-turbo-0613"
    assert turbo_model.organization is None

    sotu_store = lloom.stores["sotu_db"]
    assert sotu_store.collection.name == "sotu"
    assert isinstance(sotu_store, ChromaClient)

    count = ChromaClient(
        collection_name="sotu", persistent_directory="./test_db"
    ).collection.count()
    assert count == 110

    agent = lloom.agents["helpful_assistant"]
    assert agent.input == ["context", "query"]
    assert (
        agent.prompt
        == "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}\n\nQuestion: {query}\nHelpful Answer:\n"  # noqa: E501
    )
    assert isinstance(agent.model, ChatModel)
    assert (
        agent.system_statement
        == "I am an AI with a specialty in American political science. How can I assist you today?"  # noqa: E501
    )
