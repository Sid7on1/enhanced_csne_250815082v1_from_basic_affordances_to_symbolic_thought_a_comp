import logging
import os
import yaml
from typing import Dict, List, Optional
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
CONFIG_FILE = 'config.yaml'
DEFAULT_CONFIG = {
    'model': {
        'name': 'default_model',
        'type': 'hebbian'
    },
    'data': {
        'path': '/path/to/data',
        'format': 'csv'
    },
    'training': {
        'batch_size': 32,
        'epochs': 10
    }
}

# Define an Enum for model types
class ModelType(Enum):
    HEBBIAN = 'hebbian'
    MACHINE = 'machine'
    LISA = 'lisa'

# Define a dataclass for model configuration
@dataclass
class ModelConfig:
    name: str
    type: ModelType

# Define a dataclass for data configuration
@dataclass
class DataConfig:
    path: str
    format: str

# Define a dataclass for training configuration
@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int

# Define a dataclass for the main configuration
@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig

# Define a function to load the configuration from a file
def load_config(file_path: Optional[str] = None) -> Config:
    if file_path is None:
        file_path = CONFIG_FILE

    if not os.path.exists(file_path):
        logger.warning(f'Config file not found: {file_path}')
        return Config(**DEFAULT_CONFIG)

    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    return Config(
        model=ModelConfig(**config['model']),
        data=DataConfig(**config['data']),
        training=TrainingConfig(**config['training'])
    )

# Define a function to save the configuration to a file
def save_config(config: Config, file_path: Optional[str] = None) -> None:
    if file_path is None:
        file_path = CONFIG_FILE

    config_dict = {
        'model': {
            'name': config.model.name,
            'type': config.model.type.value
        },
        'data': {
            'path': config.data.path,
            'format': config.data.format
        },
        'training': {
            'batch_size': config.training.batch_size,
            'epochs': config.training.epochs
        }
    }

    with open(file_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

# Define a function to validate the configuration
def validate_config(config: Config) -> None:
    if config.model.type not in [ModelType.HEBBIAN, ModelType.MACHINE, ModelType.LISA]:
        raise ValueError('Invalid model type')

    if not os.path.exists(config.data.path):
        raise ValueError('Data path does not exist')

    if config.training.batch_size <= 0:
        raise ValueError('Batch size must be positive')

    if config.training.epochs <= 0:
        raise ValueError('Epochs must be positive')

# Define a context manager for loading and saving the configuration
@contextmanager
def config_context(file_path: Optional[str] = None) -> Config:
    config = load_config(file_path)
    try:
        yield config
    finally:
        save_config(config, file_path)

# Define a function to get the configuration
def get_config() -> Config:
    return load_config()

# Define a function to set the configuration
def set_config(config: Config) -> None:
    save_config(config)

# Define a function to validate and save the configuration
def validate_and_save_config(config: Config) -> None:
    validate_config(config)
    save_config(config)