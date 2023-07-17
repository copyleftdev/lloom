import json
import uuid
from dataclasses import dataclass
from typing import Any

from .chroma import ChromaClient


class Document:
    id: str = None
    document: str = None

    def __init__(
        self,
        collection_name: str,
        id: str = None,
        document: str = None,
        persistent_dir: str = "./db",
        **kwargs,
    ):
        self.id = id
        self.document = document
        self._update_attributes(kwargs)

        self.chroma_client = ChromaClient(
            collection_name=collection_name, persistent_directory=persistent_dir
        )

        if self.id is None:
            self.id = str(uuid.uuid4())
            assert self.document is not None, "Document must be provided if id is not"
            self.chroma_client.collection.add(**self.to_chroma())
            self.chroma_client.persist()
        else:
            get_result = self.chroma_client.collection.get(ids=self.id)
            self.id = get_result["ids"][0]
            self.document = get_result["documents"][0]

    def delete(self):
        self.chroma_client.collection.delete(ids=self.id)

    @classmethod
    def get(cls, where: dict = None, where_document: dict = None, **kwds):
        return Corpus().get(
            where=where, where_document=where_document, limit=1, **kwds
        )[0]

    def update(
        self,
        document: str = None,
    ):
        if document is not None:
            self.document = document
        self.chroma_client.collection.update(**self.to_chroma())
        self.chroma_client.persist()

    def to_chroma(self):
        _id = self.id
        document = self.document
        metadata = dict(self.__dict__)
        metadata.pop("chroma_client")
        metadata.pop("id")
        metadata.pop("document")
        if len(metadata) == 0:
            metadata = None
        return {"ids": _id, "documents": document, "metadatas": metadata}

    def __repr__(self):
        return f"""
        Document(
            id={self.id},
            document={self.document},)
        """

    def _update_attributes(self, kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)


@dataclass
class Corpus:
    collection_name: str
    query_results: dict = None
    get_results: dict = None
    result_index: int = 0
    persistent_dir: str = "./db"
    documents: list = None

    def __post_init__(self):
        self.chroma_client = ChromaClient(
            collection_name=self.collection_name,
            persistent_directory=self.persistent_dir,
        )

        if self.documents is None:
            self.documents = []

    def get(
        self,
        ids: list = None,
        where: dict = None,
        where_document: dict = None,
        **kwds: Any,
    ):
        if where is None:
            where = {}
        if ids is None:
            ids = []
        get_results = self.chroma_client.collection.get(
            ids=ids, where=where, where_document=where_document, include=[], **kwds
        )
        _id_l = get_results["ids"]
        self._process_results(_id_l)
        return self

    def _process_results(self, _id_l):
        self.documents = []
        for _id in _id_l:
            self.documents.append(
                Document(
                    id=_id,
                    collection_name=self.collection_name,
                    persistent_dir=self.persistent_dir,
                )
            )

    def query(
        self,
        query_texts: str,
        n_results: int = 5,
        where: dict = None,
        result_index: int = 0,
    ):
        if where is None:
            where = {}
        query_results = self.chroma_client.collection.query(
            query_texts=query_texts, n_results=n_results, where=where, include=[]
        )
        _id_l = query_results["ids"][result_index]

        self._process_results(_id_l)
        return self

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]

    def __iter__(self):
        return iter(self.documents)

    def __str__(self):
        return "\n\n".join([document.document for document in self.documents])

    def append(self, document: Document):
        self.documents.append(document)

    def to_dict(self):
        return [
            {k: v for k, v in document.__dict__.items() if k != "chroma_client"}
            for document in self.documents
        ]

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame(self.to_dict())

    def merge_documents(self):
        return self.__str__()
