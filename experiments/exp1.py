from pyhardgen.train import train_decoder
from pyhardgen.config import load_config
from pyhardgen.datagen import generate_data_from_z
from pyhardgen.nn.decoder import Decoder
from pyhardgen.dataset import *
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    data_path = Path('../data/')
    coordinates_path = data_path / 'coordinates.csv'
    dataset_path = data_path / 'sjc_internacao.csv'
    config_path = Path('../configs/exp1.yaml')
    save_path = Path('./models/exp1')
    save_path.mkdir(parents=True, exist_ok=True)
    decoder_path = save_path / 'decoder.pt'

    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() and (config.ngpu > 0) else 'cpu')

    data = pd.read_csv(dataset_path)
    cats = data.drop(columns=["Age"]).columns.to_list()
    conts = ["Age"]
    dataset = ProblemDataset(dataset_path, coordinates_path, cats, conts)
    dataloader = create_dataloader(dataset, config.dataset.batch_size, num_workers=config.workers)
    cat_dict = {cat: len(data[cat].unique()) for cat in cats}
    n_cat = sum(cat_dict.values())
    n_cont = len(conts)
    means = data[conts].mean().to_numpy()
    stds = data[conts].std().to_numpy()
    low = (data[conts].min().to_numpy() - means) / stds
    high = (data[conts].max().to_numpy() - means) / stds

    decoder = Decoder(2, n_cont, n_cat, torch.tensor(low).to(device), torch.tensor(high).to(device), config.nn)
    if decoder_path.exists():
        decoder.load_state_dict(torch.load(decoder_path))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    writer = SummaryWriter("./logs/exp1/" + timestamp)
    
    if not decoder_path.exists():
        decoder_training = train_decoder(
            dataloader,
            decoder,
            config,
            device,
            writer,
            cat_dict
        )
        torch.save(decoder_training.model.state_dict(), save_path / 'decoder.pt')

    z1_limits = np.array([-2.5, -1.5]).reshape((2, 1))
    z2_limits = np.array([-1.0, 0.0]).reshape((2, 1))
    z_limits = np.hstack((z1_limits, z2_limits))
    columns = data.columns.to_list()
    # df = generate_data_from_z(decoder, z_limits, columns, device=device, n_samples=1000)
    # df.to_csv(data_path / 'generated_data.csv', index=False)


if __name__ == '__main__':
    main()