from pyhardgen.train import train_decoder_encoder
from pyhardgen.config import load_exp, load_measures
from pyhardgen.datagen import generate_data_from_z
from pyhardgen.measures import calculate_measures
from pyhardgen.nn.decoder import Decoder
from pyhardgen.nn.encoder import Encoder
from pyhardgen.nn.decoder_encoder import DecoderEncoder
from pyhardgen.dataset import *
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from pyhard.measures import ClassificationMeasures
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from fastai.tabular.all import *
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=False, help='Train the model')
parser.add_argument('exp', type=int, help='Experiment number')

def main():
    args = parser.parse_args()
    exp_num = args.exp
    data_path = Path('./pyhard/sjc_internacao')
    coordinates_path = data_path / 'coordinates.csv'
    matrix_path = data_path / 'projection_matrix.csv'
    pyhard_folder_path = Path(f'./pyhard/exp{exp_num}')
    pyhard_folder_path.mkdir(parents=True, exist_ok=True)
    exp_config_path = Path(f'./exp{exp_num}/exp{exp_num}.yaml')
    save_path = Path(f'./models/exp{exp_num}')
    save_path.mkdir(parents=True, exist_ok=True)
    decoder_path = save_path / 'decoder.pt'
    encoder_path = save_path / 'encoder.pt'

    config = load_exp(exp_config_path)
    dataset_path = data_path / config.dataset.filename
    device = torch.device('cuda' if torch.cuda.is_available() and (config.ngpu > 0) else 'cpu')

    data = pd.read_csv(dataset_path)
    data.drop(columns=['Row'], inplace=True)
    measures = data.columns.to_list()
    coordinates = pd.read_csv(coordinates_path)
    coordinates.drop(columns=['Row'], inplace=True)

    means_data = data.mean()
    stds_data = data.std()
    means_y = coordinates.mean()
    stds_y = coordinates.std()
    data = (data - means_data) / stds_data
    coordinates = (coordinates - means_y) / stds_y
    low = (data.min() - means_data) / stds_data
    high = (data.max() - means_data) / stds_data
    low_y = (coordinates.min() - means_y) / stds_y
    high_y = (coordinates.max() - means_y) / stds_y
   
    dataset = ProblemDataset(data, coordinates)
    dataloader = create_dataloader(dataset, config.dataset.batch_size, num_workers=config.workers)

    low_t = torch.tensor(low).to(device)
    high_t = torch.tensor(high).to(device)
    decoder = Decoder(2, dataset.n_meta_features, low_t, high_t, config.nn).to(device)
    if decoder_path.exists():
        decoder.load_state_dict(torch.load(decoder_path))
        decoder.type(torch.float64)

    low_yt = torch.tensor(low_y).to(device)
    high_yt = torch.tensor(high_y).to(device)
    encoder = Encoder(dataset.n_meta_features, 2, config.nn).to(device)
    if encoder_path.exists():
        encoder.load_state_dict(torch.load(encoder_path))
        encoder.type(torch.float64)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if args.train or not decoder_path.exists() or not encoder_path.exists():
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            decoder = nn.DataParallel(decoder, device_ids=list(range(config.ngpu)))
            encoder = nn.DataParallel(encoder, device_ids=list(range(config.ngpu)))
        
        writer = SummaryWriter(f"./logs/exp{exp_num}/" + timestamp)
        _ = train_decoder_encoder(
            dataloader,
            decoder,
            encoder,
            config,
            device,
            writer
        )
        torch.save(decoder.state_dict(), decoder_path)
        torch.save(encoder.state_dict(), encoder_path)

    z1_limits = np.array([-2., -1.]).reshape((2, 1))
    z2_limits = np.array([-1.0, 0.0]).reshape((2, 1))
    z_limits = np.hstack((z1_limits, z2_limits))
    n_samples = 1000
    df, points = generate_data_from_z(decoder, z_limits, measures, means_data, stds_data, means_y, stds_y, device, n_samples)
    points_df = pd.DataFrame(points, columns=['z_1', 'z_2'])
    
    enc_points = encoder(torch.tensor(df.values, dtype=torch.float64).to(device))
    enc_points = enc_points.detach().cpu().numpy() * stds_y + means_y
    enc_points_df = pd.DataFrame(enc_points, columns=['z_1', 'z_2'])
    
    df = df * stds_data + means_data
    measures_dict = calculate_measures(df, 'target', measures)
    measures_df = pd.DataFrame(measures_dict)
    mean_measures = measures_df.mean()
    std_measures = measures_df.std()
    measures_norm = (measures_df - mean_measures) / std_measures
    matrix = pd.read_csv(matrix_path, index_col=['Row'])
    calc_coordinates = measures_norm @ matrix.T
    all_coordinates = pd.concat([coordinates, calc_coordinates], ignore_index=True)

    df.to_csv(pyhard_folder_path / 'generated_meta_features.csv', index=False)
    points_df.to_csv(pyhard_folder_path / 'points.csv', index=False)
    enc_points_df.to_csv(pyhard_folder_path / 'enc_points.csv', index=False)
    calc_coordinates.to_csv(pyhard_folder_path / 'calc_coordinates.csv', index=False)
    all_coordinates.to_csv(pyhard_folder_path / 'all_coordinates.csv', index=False)

if __name__ == '__main__':
    main()