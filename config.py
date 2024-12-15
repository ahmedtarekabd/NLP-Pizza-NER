import os
import json
import pickle
import psutil
import numpy as np
import pandas as pd
import tqdm
from typing import Literal

#* Configurations
#* Folder Paths
DATASET_PATH = "/kaggle/input/pizza-dataset/dataset" # Kaggle
DATASET_PATH = "../../data/dataset" # Local

OUTPUT_ROOT_PATH = "/kaggle/working" # Kaggle
OUTPUT_ROOT_PATH = "../../data/saved" # Local

PROCESSED_DATA_PATH = OUTPUT_ROOT_PATH + "/data"
FEATURES_PATH = OUTPUT_ROOT_PATH + "/features"
MODELS_PATH = OUTPUT_ROOT_PATH + "/models"

#* Common Variables
token_pattern=r"(?u)\b\w+(?:'\w+)?(?:-\w+)*\b"

def run_config():
    #* Pandas
    pd.set_option('display.max_colwidth', 1000) # Show all content of the cells
    # pd.reset_option('display.max_colwidth') # Undo with 
    
    #* Config tqdm for pandas
    tqdm.tqdm.pandas()

    #* Output Folders
    os.makedirs(OUTPUT_ROOT_PATH, exist_ok=True)
    os.makedirs(FEATURES_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    # os.rmdir(OUTPUT_ROOT_PATH)
    # os.rmdir(FEATURES_PATH)
    # os.rmdir(MODELS_PATH)
    # os.rmdir(PROCESSED_DATA_PATH)
