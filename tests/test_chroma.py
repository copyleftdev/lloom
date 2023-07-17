import os

from lloom.store.chroma import ChromaClient

TEST_DB_DIR = "./test_db"


def test_ChromaClient_initialization(client):
    assert isinstance(client, ChromaClient)


def test_ChromaClient_reset(client):
    if not os.path.exists(TEST_DB_DIR):
        os.makedirs(TEST_DB_DIR)
    client.reset()
    assert not os.path.exists(TEST_DB_DIR)


def test_ChromaClient_list_collections(client):
    client.set_collection("my_collection")
    collections = [coll.name for coll in client.list_collections()]
    assert "my_collection" in collections
    assert isinstance(client.list_collections(), list)


def test_ChromaClient_set_collection(client):
    client.set_collection("new_collection")
    assert client.collection.name == "new_collection"


def test_ChromaClient_delete_collection():
    client = ChromaClient(
        collection_name="collection_to_delete",
        use_openai=False,
        persistent_directory=TEST_DB_DIR,
    )
    collections = [coll.name for coll in client.list_collections()]
    assert "collection_to_delete" in collections
    client.delete_collection("collection_to_delete")
    collections = [coll.name for coll in client.list_collections()]
    assert "collection_to_delete" not in collections
    assert client.collection is None
