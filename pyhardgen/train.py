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


def train_autoencoder(
        dl: DataLoader,
        encoder: Encoder,
        decoder: Decoder,
        config: Config,
        device: torch.device,
        writer: SummaryWriter
):
    """Train the model.

    Args:
        dl (DataLoader): DataLoader object.
        encoder (Encoder): Encoder object.
        decoder (Decoder): Decoder object.
        config (Config): Config object.
        device (torch.device): Device.
    
    Returns:
        `TrainReturn`: Model and losses.
    """
    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.to(device)
    autoencoder.train()

    criterion = match_loss(config.nn.loss)

    autoencoder_losses = []

    autoencoder_optim = Adam(autoencoder.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    for epoch in range(config.epochs):
        print(f'Autoencoder Training Epoch: {epoch+1}')
        running_loss = 0.0
        last_loss = 0.0
        running_loss_epoch = 0.0
        for i, (x, _) in enumerate(dl):
            autoencoder_optim.zero_grad()

            x = torch.tensor(x, dtype=torch.float64).to(device)

            y = autoencoder(x)
            loss_autoencoder = criterion(y, x)

            loss_autoencoder.backward()
            autoencoder_optim.step()

            running_loss += loss_autoencoder.item()
            running_loss_epoch += loss_autoencoder.item()
            if i % 100 == 99:
                last_loss = running_loss / 100
                print(f'    Batch: {i+1}, Loss: {last_loss}')
                tb_x = epoch * len(dl) + i + 1
                writer.add_scalar('Autoencoder Loss', last_loss, tb_x)
                running_loss = 0.0
            
        loss_epoch = running_loss_epoch / len(dl)
        autoencoder_losses.append(loss_epoch)
        writer.add_scalar('Autoencoder Epoch Loss', loss_epoch, epoch+1)
        print(f'Epoch Loss: {loss_epoch}')

    return TrainReturn(autoencoder, autoencoder_losses)


def train_encoder(
        dl: DataLoader,
        encoder: Encoder,
        config: Config,
        device: torch.device,
        writer: SummaryWriter
):
    """Train the encoder.

    Args:
        dl (DataLoader): DataLoader object.
        encoder (Encoder): Encoder object.
        config (Config): Config object.
        device (torch.device): Device.

    Returns:
        `TrainReturn`: Model and losses.
    """
    encoder.to(device)
    encoder.train()

    criterion = match_loss(config.nn.loss)

    encoder_losses = []

    encoder_optim = Adam(encoder.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    for epoch in range(config.epochs):
        print(f'Encoder Training Epoch: {epoch+1}')
        running_loss = 0.0
        last_loss = 0.0
        running_loss_epoch = 0.0
        for i, (x, y) in enumerate(dl):
            encoder_optim.zero_grad()
            xt = torch.tensor(x, dtype=torch.float64).to(device)
            yt = torch.tensor(y, dtype=torch.float64).to(device)

            encoded = encoder(xt)
            loss_encoder = criterion(encoded, yt)

            loss_encoder.backward()
            encoder_optim.step()

            running_loss += loss_encoder.item()
            running_loss_epoch += loss_encoder.item()

            if i % 100 == 99:
                last_loss = running_loss / 100
                print(f'    Batch: {i+1} Loss: {last_loss}')
                tb_x = epoch * len(dl) + i + 1
                writer.add_scalar('Encoder Loss', last_loss, tb_x)
                running_loss = 0.0
        
        loss_epoch = running_loss_epoch / len(dl)
        encoder_losses.append(loss_epoch)
        writer.add_scalar('Encoder Epoch Loss', loss_epoch, epoch+1)
        print(f'Epoch Loss: {loss_epoch}')

    return TrainReturn(encoder, encoder_losses)


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