import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class ProblemDataset(Dataset):
    """Problem dataset.
    
    Args:
        path (`pathlib.Path`): Path to the dataset.
    """
    def __init__(self, path: Path, y: str):
        """Init method.
        
        Args:
            path (`pathlib.Path`): Path to the dataset in CSV.
            y (str): Name of the target column.
        """
        super(ProblemDataset, self).__init__()
        self.path = path
        self.data = pd.read_csv(path)
        self.y = self.data.pop(y)

    def __len__(self) -> int:
        """Length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item.
        
        Args:
            idx (int): Index.
        
        Returns:
            torch.Tensor: Item.
        """
        return self.data.iloc[idx, :], self.y.iloc[idx]


def create_dataloader(
    path: Path, 
    y: str,
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
    dataset = ProblemDataset(path, y)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    return dataloader