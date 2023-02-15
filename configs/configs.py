import argparse
import yaml
import yaml, os
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
pre_process_yaml_file = os.path.join(current_dir, "data_preprocessor.yml")
post_process_yaml_file = os.path.join(current_dir, "data_postprocessor.yml")
    
    
def get_data_preprocessor_config():
    with open(pre_process_yaml_file, "r") as f:
        data_preprocessor_config = yaml.safe_load(f)
    return data_preprocessor_config

def get_data_postprocessor_config():
    with open(post_process_yaml_file, "r") as f:
        data_postprocessor_config = yaml.safe_load(f)
    return data_postprocessor_config