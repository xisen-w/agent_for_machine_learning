import os
import pandas as pd

class DatasetManager:
    def __init__(self, directory: str):
        self.directory = directory
        self.datasets = {}

    def load_datasets(self) -> None:
        # Load all JSON and Parquet files in the given directory
        for file_name in os.listdir(self.directory):
            file_path = os.path.join(self.directory, file_name)
            if file_name.endswith(".jsonl"):
                self.datasets[file_name] = pd.read_json(file_path, lines=True)
            elif file_name.endswith(".parquet"):
                self.datasets[file_name] = pd.read_parquet(file_path)

    def get_dataset(self, file_name: str) -> pd.DataFrame:
        # Get a specific dataset by file name
        return self.datasets.get(file_name)

    def list_datasets(self) -> list:
        # List all loaded datasets
        return list(self.datasets.keys())