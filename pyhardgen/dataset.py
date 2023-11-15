import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class ProblemDataset(Dataset):
    """Problem dataset.
    
    Args:
        data_path (`pathlib.Path`): Path to the dataset.
    """
    def __init__(self, dataset: pd.DataFrame, cats: list[str], conts: list[str], ys: list[str]):
        """Init method.
        
        Args:
            data_path (`pathlib.Path`): Path to the dataset in CSV.
            coordinates_path (`pathlib.Path`): Path to the coordinates in CSV.
        """
        super(ProblemDataset, self).__init__()
        
        self.dataset = dataset
        self.y = self.dataset.loc[:, ys]
        self.dataset = dataset.drop(columns=ys)
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

def _emb_rule(n_cat: int):
    "Empirical values in fast.ai"
    return min(600, round(1.6 * n_cat**0.56))

def _emb_size_instance(cat_dict: dict[str, int], name: str):
    n_cat = cat_dict[name]
    sz = _emb_rule(n_cat)
    return (n_cat, sz)

def get_emb_sizes(cat_dict: dict[str, int]) -> list[tuple[int, int]]:
    """Get embedding sizes.
    
    Args:
        cat_dict (dict[str, int]): Dictionary of categorical variables and their cardinality.
    
    Returns:
        list[tuple[int, int]]: List of tuples of categorical variables and their embedding sizes.
    """
    return [_emb_size_instance(cat_dict, name) for name in cat_dict.keys()]
