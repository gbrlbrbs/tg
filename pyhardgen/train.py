import torch.nn.functional as F
import torch
from dataclasses import dataclass
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Generic, TypeVar
from .nn.decoder_encoder import DecoderEncoder
from .nn.encoder import Encoder
from .nn.decoder import Decoder
from .config import Config
from .loss import match_loss, LossDecoder
from .utils import decode_cats

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

    criterion = match_loss(config.nn.loss)

    decoder_losses = []

    decoder_optim = Adam(decoder.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    for epoch in range(config.epochs):
        print(f'Decoder Training Epoch: {epoch+1}')
        running_loss = 0.0
        last_loss = 0.0
        running_loss_epoch = 0.0

        for i, (x, y) in enumerate(dl):
            decoder_optim.zero_grad()

            xt = torch.tensor(x, dtype=torch.float64).to(device)
            yt = torch.tensor(y, dtype=torch.float64).to(device)

            decoded = decoder(yt)
            loss = criterion(decoded, xt)

            loss.backward()
            decoder_optim.step()

            running_loss += loss.item()
            running_loss_epoch += loss.item()

            if i % 10 == 9:
                last_loss = running_loss / 10
                print(f'    Batch: {i+1}, Loss: {last_loss}')
                tb_x = epoch * len(dl) + i + 1
                writer.add_scalar('Decoder Loss', last_loss, tb_x)
                running_loss = 0.0

        loss_epoch = running_loss_epoch / len(dl)
        decoder_losses.append(loss_epoch)
        writer.add_scalar('Decoder Epoch Loss', loss_epoch, epoch+1)
        print(f'Epoch Loss: {loss_epoch}')
    return TrainReturn(decoder, decoder_losses)

def train_encoder(
        dl: DataLoader,
        encoder: Encoder,
        config: Config,
        device: torch.device,
        writer: SummaryWriter,
):
    """
    Train the encoder.
    
    Args:
        dl (DataLoader): DataLoader object.
        encoder (Encoder): Encoder object.
        config (Config): Config object.
        device (torch.device): Device.
    """

    encoder.to(device).type(torch.float64)
    encoder.train()

    criterion = match_loss(config.nn.loss)

    encoder_losses = []

    encoder_optim = Adam(encoder.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    for epoch in range(config.epochs):
        print(f'Encoder Training Epoch: {epoch+1}')
        running_loss = 0.0
        last_loss = 0.0
        running_loss_epoch = 0.0

        for i, (x_cat, x_cont, y) in enumerate(dl):
            encoder_optim.zero_grad()

            xt_cat = torch.tensor(x_cat, dtype=torch.float64).to(device)
            xt_cont = torch.tensor(x_cont, dtype=torch.float64).to(device)
            yt = torch.tensor(y, dtype=torch.float64).to(device)

            encoded = encoder(xt_cat, xt_cont)
            loss_encoder = criterion(encoded, yt)

            loss_encoder.backward()
            encoder_optim.step()

            running_loss += loss_encoder.item()
            running_loss_epoch += loss_encoder.item()

            if i % 10 == 9:
                last_loss = running_loss / 10
                print(f'    Batch: {i+1}, Loss: {last_loss}')
                tb_x = epoch * len(dl) + i + 1
                writer.add_scalar('Encoder Loss', last_loss, tb_x)
                running_loss = 0.0

        loss_epoch = running_loss_epoch / len(dl)
        encoder_losses.append(loss_epoch)
        writer.add_scalar('Encoder Epoch Loss', loss_epoch, epoch+1)
        print(f'Epoch Loss: {loss_epoch}')

    return TrainReturn(encoder, encoder_losses)


def train_decoder_encoder(
        dl: DataLoader,
        decoder: Decoder,
        encoder: Encoder,
        config: Config,
        device: torch.device,
        writer: SummaryWriter,
):
    """Train the decoder encoder.

    Args:
        dl (DataLoader): DataLoader object.
        decoder_encoder (DecoderEncoder): DecoderEncoder object.
        config (Config): Config object.
        device (torch.device): Device.
        cat_dict (dict[str, int]): Dictionary of categorical variables and their cardinality.

    Returns:
        `TrainReturn`: Model and losses.
    """
    decoder.type(torch.float64)
    encoder.type(torch.float64)
    decoder.train()
    encoder.train()

    criterion = match_loss(config.nn.loss)

    encoder_losses = []

    de_optim = Adam([{
        'params': decoder.parameters()
        }, {
        'params': encoder.parameters()
        }], 
        lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay
    )

    for epoch in range(config.epochs):
        print(f'Decoder-Encoder Training Epoch: {epoch+1}')
        running_loss = 0.0
        running_loss_decoder = 0.0
        running_loss_encoder = 0.0
        last_loss = 0.0
        last_loss_decoder = 0.0
        last_loss_encoder = 0.0
        running_loss_epoch = 0.0
        running_loss_epoch_decoder = 0.0
        running_loss_epoch_encoder = 0.0
        for i, (x, y) in enumerate(dl):
            de_optim.zero_grad()
            yt = torch.tensor(y, dtype=torch.float64).to(device)
            xt = torch.tensor(x, dtype=torch.float64).to(device)

            decoded = decoder(yt)
            loss_dec = criterion(decoded, xt)
            encoded = encoder(decoded)
            loss_enc = criterion(encoded, yt)
            
            loss = loss_dec + loss_enc
            loss.backward()
            de_optim.step()

            running_loss_decoder += loss_dec.item()
            running_loss_epoch_decoder += loss_dec.item()

            running_loss_encoder += loss_enc.item()
            running_loss_epoch_encoder += loss_enc.item()
            
            running_loss += loss.item()
            running_loss_epoch += loss.item()

            if i % 10 == 9:
                last_loss_decoder = running_loss_decoder / 10
                last_loss_encoder = running_loss_encoder / 10
                last_loss = running_loss / 10
                print(f'    Batch: {i+1}, Loss: {last_loss}')
                tb_x = epoch * len(dl) + i + 1
                writer.add_scalar('Decoder Loss', last_loss_decoder, tb_x)
                writer.add_scalar('Encoder Loss', last_loss_encoder, tb_x)
                writer.add_scalar('Decoder-Encoder Loss', last_loss, tb_x)
                running_loss_decoder = 0.0
                running_loss_encoder = 0.0
                running_loss = 0.0

        loss_epoch_decoder = running_loss_epoch_decoder / len(dl)
        loss_epoch_encoder = running_loss_epoch_encoder / len(dl)
        loss_epoch = running_loss_epoch / len(dl)
        encoder_losses.append(loss_epoch_encoder)
        writer.add_scalar('Decoder Epoch Loss', loss_epoch_decoder, epoch+1)
        writer.add_scalar('Encoder Epoch Loss', loss_epoch_encoder, epoch+1)
        writer.add_scalar('Decoder-Encoder Epoch Loss', loss_epoch, epoch+1)
        print(f'Epoch Loss: {loss_epoch}')

    decoder_encoder = DecoderEncoder(decoder, encoder)

    return TrainReturn(decoder_encoder, encoder_losses)

        