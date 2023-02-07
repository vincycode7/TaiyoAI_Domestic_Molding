import argparse
import yaml
from . import data_preprocessor_config, data_postprocessor_config

def get_data_preprocessor_config():
    return data_preprocessor_config

def get_data_postprocessor_config():
    return data_postprocessor_config