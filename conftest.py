import os
import pytest

from lloom.store.chroma import ChromaClient

TEST_DB_DIR = "./test_db"


@pytest.fixture(scope="session")
def client():
    if not os.path.exists(TEST_DB_DIR):
        os.makedirs(TEST_DB_DIR)
    yield ChromaClient(
        collection_name="test_collection",
        use_openai=False,
        persistent_directory=TEST_DB_DIR,
    )
