import pandas as pd
import numpy as np
from pathlib import Path

def main():
    generated_data_path = Path('../pyhard/exp1/')
    data = pd.read_csv(generated_data_path / 'data.csv')
    coordinates = pd.read_csv(generated_data_path / 'coordinates.csv')

    