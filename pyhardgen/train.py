import torch.nn.functional as F
import torch
from dataclasses import dataclass
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Generic, TypeVar
from .nn.encoder import Encoder
from .nn.decoder import Decoder
from .nn.autoencoder import Autoencoder
from .config import Config
from .loss import match_loss

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
        for i, (x, _) in enumerate(dl):
            x = torch.tensor(x).to(device)

            y = autoencoder(x)
            loss_autoencoder = criterion(y, x)

            autoencoder_losses.append(loss_autoencoder.item())

            autoencoder.zero_grad()

            autoencoder_optim.step()

            loss_autoencoder.backward()

            if i % 100 == 0:
                print(f'Autoencoder Training Epoch: {epoch}, Losses: {loss_autoencoder.item()}')

    return TrainReturn(autoencoder, autoencoder_losses)


def train_encoder(
        dl: DataLoader,
        encoder: Encoder,
        config: Config,
        device: torch.device,
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
        for i, (x, y) in enumerate(dl):
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            encoded = encoder(x)
            loss_encoder = criterion(encoded, y)
            encoder_losses.append(loss_encoder.item())

            encoder.zero_grad()

            encoder_optim.step()

            loss_encoder.backward()

            if i % 100 == 0:
                print(f'Encoder Training Epoch: {epoch}, Loss: {loss_encoder.item()}')

    return TrainReturn(encoder, encoder_losses)


def train_decoder(
        dl: DataLoader,
        decoder: Decoder,
        config: Config,
        device: torch.device,
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
    decoder.to(device)
    decoder.train()

    criterion = match_loss(config.nn.loss)

    decoder_losses = []

    decoder_optim = Adam(decoder.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    for epoch in range(config.epochs):
        for i, (x, y) in enumerate(dl):
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            decoded = decoder(y)
            loss_decoder = criterion(decoded, x)
            decoder_losses.append(loss_decoder.item())

            decoder.zero_grad()

            decoder_optim.step()

            loss_decoder.backward()

            if i % 100 == 0:
                print(f'Decoder Training Epoch: {epoch}, Loss: {loss_decoder.item()}')

    return TrainReturn(decoder, decoder_losses)