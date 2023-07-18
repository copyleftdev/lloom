from abc import ABC, abstractmethod

import tiktoken

from ..store.document import Document


class Dataset(ABC):
    def __init__(
        self,
        source: str,
        tokens_per_document: int,
        token_overlap: int,
        collection_name: str,
        encoding_name: str = "cl100k_base",
        persistent_directory: str = "./db",
        use_openai: bool = False,
    ):
        self.source = source
        self.tokens_per_document = tokens_per_document
        self.token_overlap = token_overlap
        self.encoding_name = encoding_name
        self.collection_name = collection_name
        self.persistent_directory = persistent_directory
        self.use_openai = use_openai

    @abstractmethod
    def load(self):
        pass

    def _load_document(self, file_path: str) -> Document:
        with open(file_path, "r") as file:
            document_text = file.read()

        token_count = self.num_tokens_from_string(
            document_text,
        )

        document_texts = []
        start_index = 0
        while start_index < token_count:
            end_index = start_index + self.tokens_per_document
            document_texts.append(document_text[start_index:end_index])
            start_index += self.tokens_per_document - self.token_overlap

        ids = []
        for text in document_texts:
            document = Document(
                document=text,
                collection_name=self.collection_name,
                persistent_directory=self.persistent_directory,
            )
            ids.append(document.id)

        return ids

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(self.encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
