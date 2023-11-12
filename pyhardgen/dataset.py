import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class ProblemDataset(Dataset):
    """Problem dataset.
    
    Args:
        data_path (`pathlib.Path`): Path to the dataset.
    """
    def __init__(self, data: pd.DataFrame, coordinates_path: Path, cats: list[str], conts: list[str]):
        """Init method.
        
        Args:
            data_path (`pathlib.Path`): Path to the dataset in CSV.
            coordinates_path (`pathlib.Path`): Path to the coordinates in CSV.
        """
        super(ProblemDataset, self).__init__()
        
        coordinates = pd.read_csv(coordinates_path)
        self.dataset = data.merge(coordinates, left_index=True, right_index=True)
        self.dataset.drop(columns=['Row'], inplace=True)
        self.y = self.dataset.loc[:, ['z_1', 'z_2']]
        self.dataset.drop(columns=['z_1', 'z_2'], inplace=True)
        self.cats = cats
        self.conts = conts
        self.n_features = len(self.dataset.columns)

    def __len__(self) -> int:
        """Length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """Get item.
        
        Args:
            idx (int): Index.
        
        Returns:
            tuple: Inputs and outputs.
        """
        inputs_cats = self.dataset.loc[:, self.cats].iloc[idx, :].values
        inputs_conts = self.dataset.loc[:, self.conts].iloc[idx, :].values
        outputs = self.y.iloc[idx, :].values
        return inputs_cats, inputs_conts, outputs


def create_dataloader(
    dataset: ProblemDataset,
    batch_size: int, 
    shuffle: bool = True, 
    num_workers: int = 0, 
    pin_memory: bool = False
) -> DataLoader:
    """Create a dataloader from a path.
    
    Args:
        dataset (`ProblemDataset`): Dataset.
        shuffle (bool, optional): Shuffle the dataset. Defaults to True.
        num_workers (int, optional): Number of workers. Defaults to 0.
        pin_memory (bool, optional): Pin memory. Defaults to False.
    
    Returns:
        torch.utils.data.DataLoader: Dataloader.
    """
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    return dataloader