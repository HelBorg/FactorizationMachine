import os.path

from src.utils import TransformData

RAW_DATA_PATH = "data/combined_data_1.txt"
PROCESSED_DATA_PATH = "data/processed_data.csv"
CHUNK_SIZE = 10000

if __name__ == "__main__":
    data_transform = TransformData(RAW_DATA_PATH)
    if ~os.path.exists(PROCESSED_DATA_PATH):
        data_transform.transform()
        data_transform.save_data(PROCESSED_DATA_PATH)
    else:
        data_transform.load_data(PROCESSED_DATA_PATH)
