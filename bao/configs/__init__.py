import os
import yaml
from pathlib import Path
from easydict import EasyDict as edict

with open(os.path.join(os.path.dirname(__file__), 'dataset_configs.yaml'), 'r') as file:
    dataset_config = edict(yaml.safe_load(file))

default_data_dir = os.path.join(Path(os.path.dirname(__file__)).parent.parent, 'data')