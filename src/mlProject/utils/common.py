import os
from box.exceptions import BoxValueError
import yaml
from src.mlProject import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import pandas as pd
from datetime import datetime
import numpy as np
import dill




@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

@ensure_annotations
def convert_to_datetime(data: pd.DataFrame = None, column: str = None):

    dummy = data.copy()
    dummy[column] = pd.to_datetime(dummy[column], format='%Y-%m-%d %H:%M:%S')
    return dummy

@ensure_annotations
def convert_timestamp_to_hourly(data: pd.DataFrame = None, column: str = None):
    
    dummy = data.copy()
    new_ts = dummy[column].tolist()
    new_ts = [i.strftime('%Y-%m-%d %H:00:00') for i in new_ts]
    new_ts = [datetime.strptime(i, '%Y-%m-%d %H:00:00') for i in new_ts]
    dummy[column] = new_ts
    return dummy

  

@ensure_annotations
def save_numpy_array(data_array, directory, filename):
    """
    Save a NumPy array to a specific directory.

    Parameters:
    - data_array: NumPy array to be saved.
    - directory: Directory path where the array will be saved.
    - filename: Name of the file to be created.

    Returns:
    - Full path to the saved file.
    """
    # Ensure the directory exists; create it if it doesn't
    os.makedirs(directory, exist_ok=True)

    # Join the directory path and the filename to get the full file path
    file_path = os.path.join(directory, filename)

    # Save the NumPy array to the specified file path
    np.save(file_path, data_array)

    return file_path

@ensure_annotations
def save_preprocessor(preprocessor, directory_path):
    """
    Save a preprocessor object to the specified directory.

    Parameters:
    - preprocessor: The preprocessor object to be saved.
    - directory_path: The directory path where the preprocessor object will be saved.
    """
    # Check if the directory exists, and create it if not
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Construct the full file path for saving the preprocessor
    file_path = os.path.join(directory_path, 'preprocessor_model.joblib')

    # Save the preprocessor object to the specified file path
    joblib.dump(preprocessor, file_path)

@ensure_annotations
def load_numpy_array_data(file_path):
    """
    Load a NumPy array from the specified file path.

    Parameters:
    - file_path: The path to the file containing the NumPy array.

    Returns:
    - numpy_array: The loaded NumPy array.
    """
    try:
        # Load the NumPy array from the specified file path
        numpy_array = np.load(file_path)
        return numpy_array
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"Error: Unable to load data from '{file_path}'.\n{e}")


@ensure_annotations
def load_object(file_path):
    """
    Load an object from the specified file path.

    Parameters:
    - file_path: The path to the file containing the object.

    Returns:
    - loaded_object: The loaded object.
    """
    try:
        # Load the object from the specified file path
        loaded_object = joblib.load(file_path)
        return loaded_object
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"Error: Unable to load object from '{file_path}'.\n{e}")