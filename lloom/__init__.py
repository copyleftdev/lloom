import chromadb
import yaml

from .agent import Agent
from .dataset.textfile_dataset import TextfileDataset


def parse_path_or_data(yaml_data=None, file_path=None):
    if file_path:
        with open(file_path, "r") as file:
            yaml_data = file.read()
    return yaml.safe_load(yaml_data)


class Parser:
    def __init__(self, yaml_data=None, file_path=None):
        assert (
            yaml_data is not None or file_path is not None
        ), "Must provide yaml_data or file_path"
        self.data = parse_path_or_data(yaml_data=yaml_data, file_path=file_path)


class Migration(Parser):
    def __init__(self, yaml_data=None, file_path=None):
        super().__init__(yaml_data=yaml_data, file_path=file_path)

        entities = self.data.get("entities", {})
        stores_data = entities.get("stores", {})
        datasets_data = entities.get("datasets", {})

        self.stores = self._load_stores(stores_data)
        self.datasets = self._load_dataset(datasets_data, self.stores)

    def run_migration(self):
        for dataset in self.datasets.values():
            dataset.load()

        return self.stores, self.datasets

    def _load_dataset(self, datasets_data, store_objects):
        dataset_objects = {}
        for dataset_name, dataset_info in datasets_data.items():
            collection = store_objects[dataset_info["store"]]

            if dataset_info["format"] == "txt":
                dataset = TextfileDataset(
                    source=dataset_info["source"],
                    tokens_per_document=dataset_info["tokens_per_document"],
                    token_overlap=dataset_info["token_overlap"],
                    collection=collection,
                )
            dataset_objects[dataset_name] = dataset

        return dataset_objects

    def _load_stores(self, stores_data):
        in_memory = sum(
            [store.get("in_memory", False) for store in stores_data.values()]
        )
        served = sum(
            [not store.get("in_memory", False) for store in stores_data.values()]
        )

        if in_memory > 0:
            self.in_memory_client = chromadb.Client()
        if served > 0:
            self.remote_client = chromadb.HttpClient()

        store_objects = {}
        for store_name, store_info in stores_data.items():
            chroma_collection_args = {
                "collection_name": store_info["collection"],
            }

            if store_info["provider"] == "chroma":
                if store_info.get("in_memory", False):
                    client = self.in_memory_client
                else:
                    client = self.remote_client

                chroma_collection_args = {
                    "name": store_info["collection"],
                }
                if (
                    "embedding_model" in store_info
                    and store_info["embedding_model"] == "openai/ada"
                ):
                    import os

                    from chromadb.utils.embedding_functions import \
                        OpenAIEmbeddingFunction

                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                    openai_ef = OpenAIEmbeddingFunction(
                        api_key=openai_api_key, model_name="text-embedding-ada-002"
                    )
                    chroma_collection_args["embedding_function"] = openai_ef
                store_objects[store_name] = client.get_or_create_collection(
                    **chroma_collection_args
                )

            return store_objects


class Supervisor(Parser):
    def __init__(self, yaml_data=None, file_path=None):
        super().__init__(yaml_data, file_path)

        agents_data = self.data.get("agents", {})
        routine_data = self.data.get("routine", {})

        self.agents = self._load_agents(agents_data)
        self.routine = self._load_routine(routine_data)

    def _load_agents(self, agents_data):
        agent_objects = {}
        for agent_name, agent_info in agents_data.items():
            model = agent_info["model"]
            prompt = agent_info["prompt"]
            input = agent_info["input"]
            system_statement = agent_info.get("system_statement", None)

            agent_properties = {
                "name": model,
                "prompt": prompt,
                "input": input,
                "system_statement": system_statement,
            }
            agent_objects[agent_name] = Agent(**agent_properties)
        return agent_objects

    def _load_routine(self, routine_data):
        self.trigger = routine_data["trigger"]
        self.steps = routine_data["steps"]


class Lloom(Migration, Supervisor):
    def __init__(
        self,
        yaml_data=None,
        file_path=None,
        datasets=None,
        stores=None,
        perform_migration=False,
    ):
        super().__init__(yaml_data=yaml_data, file_path=file_path)

        metadata_data = self.data.get("metadata", {})

        self._parse_metadata(metadata_data)

        if perform_migration:
            self.run_migration()

    def _parse_metadata(self, metadata_data):
        for key, value in metadata_data.items():
            setattr(self, key, value)

    def run(self, trigger_value):
        for step in self.steps:
            if step["name"] == "retrieve_relevant_documents":
                collection_name = step["store"]
                collection = self.stores[collection_name]
                results = collection.query(
                    query_texts=trigger_value,
                    include=["documents"],
                    n_results=5,
                )
            if step["name"] == "chat":
                agent_name = step["agent"]
                agent = self.agents[agent_name]

                values = {
                    "context": "\n".join(results["documents"][0]),
                    "query": trigger_value,
                }
                prompt = agent.prepare_prompt(values)
                output = agent.generate_response(prompt)
        return output
