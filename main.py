import os
import warnings
warnings.simplefilter('ignore')
from transformers import logging
logging.set_verbosity_error()
import torch
import numpy as np
import argparse
from utils.util import load_config
from accelerate.utils import set_seed
from datasets import load_dataset
from data.dataset_loader import get_datasets
from tqdm import tqdm

from inference.generate import generate
from inference.generate_vllm import generate_vllm

from generator.vanilla_trainer import train
from reward_model.orm.orm_trainer import orm_classification_trainer
from reward_model.prm.prm_trainer import prm_classification_trainer
from reward_model.orm.orm_guide import orm_guided_generation
from reward_model.prm.prm_guide import prm_guided_generation

def set_seeds(seed):
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run(config):
    generate(config["generator"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/bioasq.yaml', type=str, help='Path to the config file')
    parser.add_argument('--debug', default='generation', type=str, help='debug')
    args = parser.parse_args()

    config_path = args.config
    assert os.path.isfile(config_path), f"Invalid config path: {config_path}"

    config = load_config(config_path)

    # set seeds
    set_seeds(config['seed'])
    if args.debug == 'generation':
        generate(config["generator"])
    elif args.debug == 'reward':
        if 'orm' in config['reward_model']['type']:
            orm_classification_trainer(config)
        elif 'prm' in config['reward_model']['type']:
            prm_classification_trainer(config)
    elif args.debug == 'train_generator':
        train(config["generator_trainer"])
        generate(config["generator"])
    elif args.debug == 'reward_guide':
        if 'orm' in config['reward_model']['type']:
            generator = orm_guided_generation(config)
        elif 'prm' in config['reward_model']['type']:
            generator = prm_guided_generation(config)
        generator.guide_generation()