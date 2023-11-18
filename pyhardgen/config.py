from dataclasses import dataclass
from serde import serde, field, from_dict, SerdeError
from serde.yaml import from_yaml
from schema import Schema, And, Optional
from pathlib import Path
import typing as t
import yaml

@serde
@dataclass
class model:
    n_layers: int
    features: list[int]
    activation: str
    dropout: float
    loss: str

@serde
@dataclass
class dataset:
    filename: str
    batch_size: int

@serde
@dataclass
class optimizer:
    lr: float
    weight_decay: t.Optional[float] = None

@serde
@dataclass
class Config:
    name: str
    ngpu: int
    epochs: int
    runs: int
    workers: int
    nn: model
    dataset: dataset
    optimizer: optimizer

@serde
@dataclass
class MeasureList:
    measures: list[str] = field(rename='list')


class ModelSchema(Schema):

    def validate(self, data, _is_event_schema=True):
        data = super(ModelSchema, self).validate(data, _is_event_schema=False)
        if _is_event_schema and data['n_layers'] != len(data['features']):
            raise ValueError('num_layers must be equal to the length of features')
        return data


SCHEMA = Schema({
    'name': And(str, len),
    'ngpu': And(int, lambda n: 0 <= n),
    'epochs': And(int, lambda n: 0 < n),
    'runs': And(int, lambda n: 0 < n),
    'workers': And(int, lambda n: 0 <= n),
    'nn': ModelSchema({
        'n_layers': And(int, lambda n: 0 < n),
        'features': And([And(int, lambda n: 0 < n)], len),
        'activation': And(str, len),
        'dropout': And(float, lambda n: 0 <= n < 1),
        'loss': And(str, len),
    }),
    'dataset': {
        'filename': And(str, len),
        'batch_size': And(int, lambda n: 0 < n),
    },
    'optimizer': {
        'lr': And(float, lambda n: 0 < n),
        Optional('weight_decay'): And(float, lambda n: 0 <= n),
    },
})

def load_exp(path: Path) -> Config:
    with open(path, 'r') as file:
        exp_raw_str = file.read()

    exp_raw = yaml.safe_load(exp_raw_str)
    
    SCHEMA.validate(exp_raw)

    exp = from_yaml(Config, exp_raw_str)

    return exp

def load_measures(path: Path) -> MeasureList:
    with open(path, 'r') as file:
        config_raw_str = file.read()

    config_raw = yaml.safe_load(config_raw_str)
    measures_raw = config_raw['measures']
    try: 
        measures = from_dict(MeasureList, measures_raw)
        return measures
    except SerdeError as e:
        print(e)
        raise e