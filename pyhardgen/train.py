import torch.nn.functional as F
import torch
from dataclasses import dataclass
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Generic, TypeVar
from .nn.encoder import Encoder
from .nn.decoder import Decoder
from .nn.autoencoder import Autoencoder
from .config import Config
from .loss import match_loss, LossDecoder

T = TypeVar('T')

@dataclass
class TrainReturn(Generic[T]):
    model: T
    losses: list[float]


def train_decoder(
        dl: DataLoader,
        decoder: Decoder,
        config: Config,
        device: torch.device,
        writer: SummaryWriter,
        cat_dict: dict[str, int]
):
    """Train the decoder.

    Args:
        dl (DataLoader): DataLoader object.
        decoder (Decoder): Decoder object.
        config (Config): Config object.
        device (torch.device): Device.


    Returns:
        `TrainReturn`: Model and losses.
    """
    decoder.to(device).type(torch.float64)
    decoder.train()

    criterion = LossDecoder(cat_dict)

    decoder_losses = []

    decoder_optim = Adam(decoder.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    for epoch in range(config.epochs):
        print(f'Decoder Training Epoch: {epoch+1}')
        running_loss = 0.0
        last_loss = 0.0
        running_loss_epoch = 0.0
        for i, (x_cat, x_cont, y) in enumerate(dl):
            decoder_optim.zero_grad()

            xt_cat = torch.tensor(x_cat, dtype=torch.float64).to(device)
            xt_cont = torch.tensor(x_cont, dtype=torch.float64).to(device)
            yt = torch.tensor(y, dtype=torch.float64).to(device)

            decoded = decoder(yt)
            loss_decoder = criterion(decoded, xt_cat, xt_cont)

            loss_decoder.backward()
            decoder_optim.step()

            running_loss += loss_decoder.item()
            running_loss_epoch += loss_decoder.item()

            if i % 20 == 19:
                last_loss = running_loss / 100
                print(f'    Batch: {i+1}, Loss: {last_loss}')
                tb_x = epoch * len(dl) + i + 1
                writer.add_scalar('Decoder Loss', last_loss, tb_x)
                running_loss = 0.0

        loss_epoch = running_loss_epoch / len(dl)
        decoder_losses.append(loss_epoch)
        writer.add_scalar('Decoder Epoch Loss', loss_epoch, epoch+1)
        print(f'Epoch Loss: {loss_epoch}')

    return TrainReturn(decoder, decoder_losses)