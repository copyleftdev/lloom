import pytest
from chromadb import Client
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from lloom.dataset.textfile_dataset import TextfileDataset


@pytest.fixture(scope="session")
def client():
    return Client()


@pytest.fixture(scope="function")
def test_collection(client):
    collection = client.get_or_create_collection(
        name="test_collection",
    )
    yield collection
    collection.delete()


@pytest.fixture(scope="function")
def test_collection_openai(client):
    import os

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_ef = OpenAIEmbeddingFunction(
        api_key=openai_api_key, model_name="text-embedding-ada-002"
    )
    collection = client.get_or_create_collection(
        name="test_collection_openai",
        embedding_function=openai_ef,
    )
    yield collection
    collection.delete()


def test_load_textfile_dataset(test_collection):
    dataset = TextfileDataset(
        source="./tests/sotu.txt",
        tokens_per_document=1000,
        token_overlap=100,
        collection=test_collection,
    )

    ids = dataset.load()
    assert len(ids) == 47
    doc = test_collection.get(ids=ids[15])["documents"].pop()
    assert (
        doc.split("\n")[0] == "Roads and water systems to withstand the next big flood."
    )


def test_load_textfile_dataset_openai(test_collection_openai):
    dataset = TextfileDataset(
        source="./tests/sotu-small.txt",
        tokens_per_document=1000,
        token_overlap=100,
        collection=test_collection_openai,
    )

    ids = dataset.load()
    assert len(ids) == 1


