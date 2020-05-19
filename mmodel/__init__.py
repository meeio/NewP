import os
import sys
from importlib import import_module
from pathlib import Path

import comet_ml

from mmodel.basic_params import basic_parser


def get_basic_params():
    model = os.environ["TARGET_MODEL"]
    config_path = Path('./mmodel/{}/config.yml'.format(model))
    if not config_path.exists():
        raise Exception("{} should be config file".format(config_path))
    basic_parser._default_config_files.append(config_path)
    params, _ = basic_parser.parse_known_args()
    return params

def get_model():
    model_name = os.environ["TARGET_MODEL"]

    dir_path = Path('./mmodel/'+model_name)
    if not dir_path.exists():
        raise Exception("{} not exists.".format(model_name))
    os.environ["TARGET_MODEL"] = model_name

    model = import_module('mmodel.' + model_name).model()
    return model

