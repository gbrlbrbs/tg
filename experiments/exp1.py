from pyhardgen.train import train_encoder, train_decoder
from pyhardgen.config import load_config
from pyhardgen.datagen import generate_data_from_z
from pyhardgen.dataset import ProblemDataset
from pyhardgen.nn.encoder import Encoder
from pyhardgen.nn.decoder import Decoder
import torch
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    data_path = Path('../data/')
    coordinates_path = data_path / 'coordinates.csv'
    dataset_path = data_path / 'sjc_internacao.csv'
    config_path = Path('../configs/exp1.yaml')

    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() and (config.ngpu > 0) else 'cpu')

    dataset = ProblemDataset(dataset_path, coordinates_path)
    encoder = Encoder(dataset.n_features, 2, config.nn)
    decoder = Decoder(2, dataset.n_features, config.nn)