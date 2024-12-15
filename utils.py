import os
import json
import pickle
import pyarrow as pa
import pyarrow.dataset as pda
import pyarrow.parquet as pq
import glob
import psutil
import numpy as np
import pandas as pd
from typing import Literal

from config import FEATURES_PATH, MODELS_PATH, OUTPUT_ROOT_PATH, PROCESSED_DATA_PATH

types = Literal["model", "feature", "processed"]

#* General
def file_exists(path):
    return os.path.exists(path)


#* Memory Management & Performance
def memory_usage():
    process = psutil.Process(os.getpid())
    return (process.memory_info().rss / 1024 ** 2)


#* Save & Load functions
def save_pickle(path, obj, type: types | None = None):
    if type is not None:
        if type == "model":
            path = MODELS_PATH + "/" + path
        elif type == "feature":
            path = FEATURES_PATH + "/" + path
        elif type == "processed":
            path = PROCESSED_DATA_PATH + "/" + path
    with open (path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path, type: types | None = None):
    if type is not None:
        if type == "model":
            path = MODELS_PATH + "/" + path
        elif type == "feature":
            path = FEATURES_PATH + "/" + path
        elif type == "processed":
            path = PROCESSED_DATA_PATH + "/" + path
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def save_np(path, obj, type: types | None = None, allow_pickle=True):
    if type is not None:
        if type == "model":
            path = MODELS_PATH + "/" + path
        elif type == "feature":
            path = FEATURES_PATH + "/" + path
        elif type == "processed":
            path = PROCESSED_DATA_PATH + "/" + path
            
    np.save(path, obj, allow_pickle=allow_pickle)

def load_np(path, type: types | None = None, allow_pickle=True):
    if type is not None:
        if type == "model":
            path = MODELS_PATH + "/" + path
        elif type == "feature":
            path = FEATURES_PATH + "/" + path
        elif type == "processed":
            path = PROCESSED_DATA_PATH + "/" + path

    return np.load(path, allow_pickle=allow_pickle)

def load_json(filename: str, cols: list[str] | None = None):
    """
    Load a json file into a pandas DataFrame.
    * This function is useful (for some reason) for loading the large dataset files.
    
    filename: str
        The name of the file to load.
    cols: list[str] | None
        The columns to load. If None, load all columns.
    return: pd.DataFrame
        The DataFrame containing the data from the json file.
    """
    all_cols = True if cols is None else False
    data = []

    with open(filename, encoding='latin-1') as f:
        line = f.readline()
        f.seek(0) # Go back to the beginning of the file
        doc = json.loads(line)
        if all_cols:
            cols = list(doc.keys())
        
        for line in f:
            doc = json.loads(line)
            lst = [doc[col] for col in cols]
            data.append(lst)

    df = pd.DataFrame(data=data, columns=cols)
    return df


def process_parquet_in_chunks(input_file: str, output_file: str, chunk_size: int, preprocess_function: callable, args: tuple = (), merge_chunks: bool=True):
    """
    Process a large Parquet file in chunks, applying a preprocessing function to each row, 
    and save the processed chunks as new Parquet files. Optionally merge the processed chunks.
    Source: https://blog.clairvoyantsoft.com/efficient-processing-of-parquet-files-in-chunks-using-pyarrow-b315cc0c62f9

    Parameters:
    - input_file (str): Path to the input Parquet file.
    - output_file (str): Path to save the processed Parquet file.
    - chunk_size (int): Number of rows to process per chunk.
    - preprocess_function (function): Function to apply to each row.
    - merge_chunks (bool): Whether to merge the processed chunks into a single Parquet file (default: True).

    Returns:
    - None
    """

    parquet_file = pq.ParquetFile(input_file) # Dataframe which does not fit into system memory

    for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
        df = batch.to_pandas()
        # Process the chunk (batch)
        processed_chunk = df.progress_apply(preprocess_function, args=args, axis=1)

        # Save the processed chunk to a new Parquet file
        output_chunk = f"{output_file}_{i}.parquet"
        processed_chunk.to_parquet(output_chunk, engine='pyarrow', compression='snappy')
        print(f"Chunk {i} processed and saved to {output_chunk}")

    # Optionally merge processed chunks
    if merge_chunks:
        print("Merging processed chunks into a single Parquet file...")

        # Get all processed chunk files
        parquet_files = glob.glob(f"{output_file}_*.parquet")
        # Read and concatenate them into a single DataFrame
        final_df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
        # Save the final DataFrame as a single Parquet file
        final_df.to_parquet(output_file, engine='pyarrow', compression='snappy')
        # Remove the processed chunk files
        for file in parquet_files:
            os.remove(file)

        print(f"Merged file saved to {output_file}")


def process_pickles_in_chunks(input_file: str, output_file: str, chunk_size: int, preprocess_function: callable, args: tuple = (), merge_chunks: bool=True):
    """
    Process a large pickle file in chunks, applying a preprocessing function to each row, 
    and save the processed chunks as new pickle files. Optionally merge the processed chunks.

    Parameters:
    - input_file (str): Path to the input pickle file.
    - output_file (str): Path to save the processed pickle file.
    - chunk_size (int): Number of rows to process per chunk.
    - preprocess_function (function): Function to apply to each row.
    - merge_chunks (bool): Whether to merge the processed chunks into a single pickle file (default: True).

    Returns:
    - None
    """

    # Load the pickle file
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    # Split the data into chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    for i, chunk in enumerate(chunks):
        # Process the chunk
        processed_chunk = [preprocess_function(row, *args) for row in chunk]

        # Save the processed chunk to a new pickle file
        output_chunk = f"{output_file}_{i}.pkl"
        with open(output_chunk, 'wb') as f:
            pickle.dump(processed_chunk, f)

        print(f"Chunk {i} processed and saved to {output_chunk}")

    # Optionally merge processed chunks
    if merge_chunks:
        print("Merging processed chunks into a single pickle file...")

        # Get all processed chunk files
        pickle_files = glob.glob(f"{output_file}_*.pkl")
        # Read and concatenate them into a single list
        final_data = []
        for file in pickle_files:
            with open(file, 'rb') as f:
                final_data.extend(pickle.load(f))
        # Save the final list as a single pickle file
        with open(output_file, 'wb') as f:
            pickle.dump(final_data, f)
        # Remove the processed chunk files
        for file in pickle_files:
            os.remove(file)

        print(f"Merged file saved to {output_file}")

