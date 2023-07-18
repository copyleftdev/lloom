from lloom import Lloom
from lloom.store.chroma import ChromaClient


def test_create_objects_from_yaml():
    file_path = "./tests/sotu.yml"
    lloom = Lloom(file_path=file_path)
    lloom.create_objects_from_yaml()

    objects = lloom.objects
    models = objects["models"]
    stores = objects["stores"]
    datasets = objects["datasets"]

    assert len(models) == 1
    assert "turbo" in models

    assert len(stores) == 1
    assert "sotu_db" in stores

    assert len(datasets) == 1
    assert "sotu_raw" in datasets

    turbo_model = models["turbo"]
    assert turbo_model.name == "gpt-3.5-turbo-0613"
    assert turbo_model.organization is None

    sotu_store = stores["sotu_db"]
    assert sotu_store.collection.name == "sotu"
    assert isinstance(sotu_store, ChromaClient)

    count = ChromaClient(
        collection_name="sotu", persistent_directory="./test_db"
    ).collection.count()
    assert count == 110
