from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    all_schema: dict
    input_folder : Path
    output_folder : Path
    input_file1 : Path
    input_file2 : Path
    input_file3 : Path
    output_directory : Path
    merged_output: Path



@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_file_path: Path
    test_file_path: Path
    transformed_object_file_path: Path
    transformed_train_file_path: Path
    transformed_test_file_path: Path



@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    target_column: str
    n_estimators: int
    min_weight_fraction_leaf: float
    min_samples_split: int
    min_samples_leaf: int
    min_impurity_decrease: float
    max_leaf_nodes: int
    max_depth: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str