import yaml

from lloom.dataset.csvfile_dataset import CSVfileDataset
from lloom.dataset.textfile_dataset import TextfileDataset
from lloom.model.openai import AdaModel, ChatModel
from lloom.store.chroma import ChromaClient


class Lloom:
    def __init__(self, yaml_data):
        self.yaml_data = yaml_data
        self.objects = []


def create_objects_from_yaml(self):
    data = yaml.safe_load(self.yaml_data)

    entities = data.get("entities", {})
    models = entities.get("models", [])
    stores = entities.get("stores", [])
    datasets = entities.get("datasets", [])

    model_objects = {}

    for model in models:
        model_name = next(iter(model))
        model_info = model[model_name]

        if model_info["kind"] == "chat":
            model_params = {
                "name": model_info["name"],
            }
            if "organization" in model_info:
                model_params["organization"] = model_info["organization"]
            chat_model = ChatModel(**model_params)
            model_objects[model_name] = chat_model
        elif model_info["kind"] == "embedding":
            model_params = {}
            if "organization" in model_info:
                model_params["organization"] = model_info["organization"]
            ada_model = AdaModel(**model_params)
            model_objects[model_name] = ada_model

    store_objects = {}

    for store in stores:
        store_name = next(iter(store))
        store_info = store[store_name]

        if store_info["provider"] == "chroma":
            chroma_client = ChromaClient(
                *store_info.get("args", []), **store_info.get("kwargs", {})
            )
            store_objects[store_name] = chroma_client

    dataset_objects = {}

    for dataset in datasets:
        dataset_name = next(iter(dataset))
        dataset_info = dataset[dataset_name]

        if dataset_info["format"] == "csv":
            csvfile_dataset = CSVfileDataset(
                source=dataset_info["source"],
                format=dataset_info["format"],
                tokens_per_document=dataset_info["tokens_per_document"],
                token_overlap=dataset_info["token_overlap"],
                text_field=dataset_info["text_field"],
                collection_name=dataset_info["store"],
            )
            dataset_objects[dataset_name] = csvfile_dataset
        elif dataset_info["format"] == "txt":
            textfile_dataset = TextfileDataset(
                source=dataset_info["source"],
                format=dataset_info["format"],
                tokens_per_document=dataset_info["tokens_per_document"],
                token_overlap=dataset_info["token_overlap"],
                collection_name=dataset_info["store"],
            )
            dataset_objects[dataset_name] = textfile_dataset

    self.objects = {
        "models": model_objects,
        "stores": store_objects,
        "datasets": dataset_objects,
    }
