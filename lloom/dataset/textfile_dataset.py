import glob

from .abc import Dataset


class TextfileDataset(Dataset):
    def load(self):
        file_paths = glob.glob(self.source)
        if not file_paths:
            raise ValueError("No files found matching the provided glob pattern.")

        ids = []
        for file_path in file_paths:
            ids += self._load_document(file_path)

        return ids
