from pyhardgen.train import train_decoder_encoder
from pyhardgen.config import load_config
from pyhardgen.datagen import generate_data_from_z
from pyhardgen.nn.decoder import Decoder
from pyhardgen.nn.encoder import Encoder
from pyhardgen.nn.decoder_encoder import DecoderEncoder
from pyhardgen.dataset import *
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import torch
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
    pyhard_folder_path = Path(f'./pyhard/exp{exp_num}')
    pyhard_folder_path.mkdir(parents=True, exist_ok=True)
    config_path = Path(f'./exp{exp_num}/exp{exp_num}.yaml')
    save_path = Path(f'./models/exp{exp_num}')
    save_path.mkdir(parents=True, exist_ok=True)
    decoder_path = save_path / 'decoder.pt'
    encoder_path = save_path / 'encoder.pt'

    config = load_config(config_path)
    dataset_path = data_path / config.dataset.filename
    device = torch.device('cuda' if torch.cuda.is_available() and (config.ngpu > 0) else 'cpu')

    data = pd.read_csv(dataset_path)
    coordinates = pd.read_csv(coordinates_path)
    conts = ["Age"]
    cats = data.drop(columns=conts).columns.to_list()
    cat_dict = {cat: len(data[cat].unique()) for cat in cats}
    data = data.merge(coordinates, left_index=True, right_index=True)
    data.drop(columns=['Row'], inplace=True)
    emb_szs = get_emb_sizes(cat_dict)
    n_cat = sum(cat_dict.values())
    n_cont = len(conts)
    means = data[conts].mean().to_numpy()
    stds = data[conts].std().to_numpy()
    low = (data[conts].min().to_numpy() - means) / stds
    high = (data[conts].max().to_numpy() - means) / stds
    data[conts] = (data[conts] - means) / stds
    ys = ['z_1', 'z_2']
    means_y = data[ys].mean().to_numpy()
    stds_y = data[ys].std().to_numpy()
    low_y = (data[ys].min().to_numpy() - means_y) / stds_y
    high_y = (data[ys].max().to_numpy() - means_y) / stds_y
    data[ys] = (data[ys] - means_y) / stds_y
    dataset = ProblemDataset(data, cats, conts, ys)
    dataloader = create_dataloader(dataset, config.dataset.batch_size, num_workers=config.workers)

    decoder = Decoder(2, n_cont, n_cat, torch.tensor(low).to(device), torch.tensor(high).to(device), config.nn).to(device)
    if decoder_path.exists():
        decoder.load_state_dict(torch.load(decoder_path))
        decoder.type(torch.float64)

    low_yt = torch.tensor(low_y).to(device)
    high_yt = torch.tensor(high_y).to(device)
    encoder = Encoder(emb_szs, n_cont, 2, config.nn, low_yt, high_yt).to(device)
    if encoder_path.exists():
        encoder.load_state_dict(torch.load(encoder_path))
        encoder.type(torch.float64)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    if args.train or not decoder_path.exists() or not encoder_path.exists():
        writer = SummaryWriter(f"./logs/exp{exp_num}/" + timestamp)
        _ = train_decoder_encoder(
            dataloader,
            decoder,
            encoder,
            config,
            device,
            writer,
            cat_dict
        )
        torch.save(decoder.state_dict(), decoder_path)
        torch.save(encoder.state_dict(), encoder_path)

    z1_limits = np.array([-2.5, -1.5]).reshape((2, 1))
    z2_limits = np.array([-1.0, 0.0]).reshape((2, 1))
    z_limits = np.hstack((z1_limits, z2_limits))
    n_samples = 1000
    df, points = generate_data_from_z(decoder, z_limits, cats, conts, means, stds, means_y, stds_y, device, cat_dict, n_samples)
    points_df = pd.DataFrame(points, columns=['z_1', 'z_2'])
    enc_points = encoder(torch.tensor(df[cats].values, dtype=torch.int).to(device), torch.tensor(df[conts].values, dtype=torch.float64).to(device))
    enc_points = enc_points.detach().cpu().numpy() * stds_y + means_y
    enc_points_df = pd.DataFrame(enc_points, columns=['z_1', 'z_2'])
    data = pd.read_csv(dataset_path)
    data = pd.concat([data, df])
    df.to_csv(pyhard_folder_path / 'generated_data.csv', index=False)
    data.to_csv(pyhard_folder_path / 'data.csv', index=False)
    points_df.to_csv(pyhard_folder_path / 'points.csv', index=False)
    enc_points_df.to_csv(pyhard_folder_path / 'enc_points.csv', index=False)

if __name__ == '__main__':
    main()