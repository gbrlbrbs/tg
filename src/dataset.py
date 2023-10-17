import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class ProblemDataset(Dataset):
    """Problem dataset.
    
    Args:
        data_path (`pathlib.Path`): Path to the dataset.
    """
    def __init__(self, data_path: Path, coordinates_path: Path):
        """Init method.
        
        Args:
            data_path (`pathlib.Path`): Path to the dataset in CSV.
            coordinates_path (`pathlib.Path`): Path to the coordinates in CSV.
        """
        super(ProblemDataset, self).__init__()
        data = pd.read_csv(data_path)
        coordinates = pd.read_csv(coordinates_path)
        self.dataset = data.merge(coordinates, left_index=True, right_index=True)
        self.dataset.drop(columns=['Row'], inplace=True)
        self.y = self.dataset.drop(columns=['z_1', 'z_2'])

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
        inputs = self.dataset.iloc[idx, :].values
        outputs = self.y.iloc[idx, :].values
        return inputs, outputs


def create_dataloader(
    data_path: Path, 
    coordinates_path: Path,
    batch_size: int, 
    shuffle: bool = True, 
    num_workers: int = 0, 
    pin_memory: bool = False
) -> DataLoader:
    """Create a dataloader from a path.
    
    Args:
        path (Path): Path to the dataset.
        batch_size (int): Batch size.
        shuffle (bool, optional): Shuffle the dataset. Defaults to True.
        num_workers (int, optional): Number of workers. Defaults to 0.
        pin_memory (bool, optional): Pin memory. Defaults to False.
    
    Returns:
        torch.utils.data.DataLoader: Dataloader.
    """
    dataset = ProblemDataset(data_path, coordinates_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    return dataloader