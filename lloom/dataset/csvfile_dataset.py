import glob

import pandas as pd

from ..store.document import Document
from .abc import Dataset


class CSVfileDataset(Dataset):
    def __init__(
        self,
        source: str,
        tokens_per_document: int,
        token_overlap: int,
        text_field: str,
        collection_name: str,
        encoding_name: str = "cl100k_base",
        persistent_directory: str = "./db",
    ):
        self.source = source
        self.tokens_per_document = tokens_per_document
        self.token_overlap = token_overlap
        self.text_field = text_field
        self.encoding_name = encoding_name
        self.collection_name = collection_name
        self.persistent_directory = persistent_directory

    def load(self):
        file_paths = glob.glob(self.source)
        if not file_paths:
            raise ValueError("No files found matching the provided glob pattern.")

        for file_path in file_paths:
            dataframe = pd.read_csv(file_path)
            if self.text_field not in dataframe.columns:
                raise ValueError(
                    f"Text field '{self.text_field}' not found in the CSV file."
                )

            text_column = dataframe[self.text_field]
            ids = []
            for document_text in text_column:
                ids += self._load_document(str(document_text))

        return ids

    def _load_document(self, document_text: str):
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
