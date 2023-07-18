import yaml

from lloom.agent import Agent
from lloom.dataset.csvfile_dataset import CSVfileDataset
from lloom.dataset.textfile_dataset import TextfileDataset
from lloom.model.openai import AdaModel, ChatModel
from lloom.store.chroma import ChromaClient


class Lloom:
    def __init__(self, yaml_data=None, file_path=None):
        if file_path:
            with open(file_path, "r") as file:
                self.yaml_data = file.read()
        else:
            self.yaml_data = yaml_data

    def create_objects_from_yaml(self):
        data = yaml.safe_load(self.yaml_data)

        entities = data.get("entities", {})
        models = entities.get("models", {})
        stores = entities.get("stores", {})
        datasets = entities.get("datasets", {})

        agents = data.get("agents", {})

        model_objects = {}
        for model_name, model_info in models.items():
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

        self.models = model_objects

        store_objects = {}
        for store_name, store_info in stores.items():
            if store_info["provider"] == "chroma":
                chroma_client_args = {
                    "collection_name": store_info["collection"],
                }
                if "persistent_directory" in store_info:
                    chroma_client_args["persistent_directory"] = store_info[
                        "persistent_directory"
                    ]
                if (
                    "embedding_model" in store_info
                    and store_info["embedding_model"] == "ada"
                ):
                    chroma_client_args["use_openai"] = True
                else:
                    chroma_client_args["use_openai"] = False

                chroma_client = ChromaClient(
                    **chroma_client_args,
                )
                store_objects[store_name] = chroma_client
        self.stores = store_objects

        dataset_objects = {}
        for dataset_name, dataset_info in datasets.items():
            store = store_objects[dataset_info["store"]]

            if dataset_info["format"] == "csv":
                dataset = CSVfileDataset(
                    source=dataset_info["source"],
                    format=dataset_info["format"],
                    tokens_per_document=dataset_info["tokens_per_document"],
                    token_overlap=dataset_info["token_overlap"],
                    text_field=dataset_info["text_field"],
                    collection_name=store.collection.name,
                    persistent_directory=store.persistent_directory,
                    use_openai=store.use_openai,
                )
            elif dataset_info["format"] == "txt":
                dataset = TextfileDataset(
                    source=dataset_info["source"],
                    tokens_per_document=dataset_info["tokens_per_document"],
                    token_overlap=dataset_info["token_overlap"],
                    collection_name=store.collection.name,
                    persistent_directory=store.persistent_directory,
                    use_openai=store.use_openai,
                )
            dataset.load()
            dataset_objects[dataset_name] = dataset
        self.datasets = dataset_objects

        agent_objects = {}
        for agent_name, agent_info in agents.items():
            model = model_objects[agent_info["model"]]
            prompt = agent_info["prompt"]
            input = agent_info["input"]
            system_statement = agent_info.get("system_statement", None)

            agent = Agent(
                name=agent_name,
                model=model,
                prompt=prompt,
                input=input,
                system_statement=system_statement,
            )
            agent_objects[agent_name] = agent
        self.agents = agent_objects
