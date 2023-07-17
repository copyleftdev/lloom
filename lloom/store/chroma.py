import os
import shutil
from dataclasses import dataclass

import chromadb
import openai
from chromadb.config import Settings
from openai import Embedding
from sentence_transformers import SentenceTransformer


def create_embeddings_openai(_input: list[str]):
    results = []
    for text in _input:
        embedding = Embedding.create(model="text-embedding-ada-002", input=text)[
            "data"
        ][0]["embedding"]
        results.append(embedding)
    return results


def create_embeddings_local(_input: list[str]):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(_input)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        collection_name = kwargs.get("collection_name")
        persistent_directory = kwargs.get("persistent_directory")
        key = (collection_name, persistent_directory)
        if key not in cls._instances:
            cls._instances[key] = super().__call__(*args, **kwargs)
        return cls._instances[key]


@dataclass
class ChromaClient(metaclass=Singleton):
    collection_name: str
    persistent_directory: str = "./db"
    use_openai: bool = False

    def __post_init__(self):
        self.chroma_client = chromadb.Client(
            settings=Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persistent_directory,
            )
        )
        if self.use_openai:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.embedding_function = create_embeddings_openai
        else:
            self.embedding_function = create_embeddings_local
        self.set_collection(self.collection_name)

        self.chroma_client.persist()

    def delete_collection(self, collection_name: str):
        if self.collection is not None and self.collection.name == collection_name:
            self.collection = None
        self.chroma_client.delete_collection(collection_name)

    def list_collections(self) -> list:
        return self.chroma_client.list_collections()

    def set_collection(self, collection_name: str):
        self.collection = self.chroma_client.get_or_create_collection(
            embedding_function=self.embedding_function,
            name=collection_name,
        )

    def persist(self):
        self.chroma_client.persist()

    def reset(self):
        shutil.rmtree(self.persistent_directory, ignore_errors=True)
