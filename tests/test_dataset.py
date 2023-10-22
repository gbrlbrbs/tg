from pyhardgen.dataset import ProblemDataset
from pathlib import Path
import unittest
import pandas as pd

data_path = Path('../data/sjc_internacao.csv')
coordinates_path = Path('../data/coordinates.csv')
problem_dataset = ProblemDataset(data_path, coordinates_path)
df = pd.read_csv(data_path)

class TestDataset(unittest.TestCase):

    def test_dataset_len(self):
        self.assertEqual(len(problem_dataset), len(df))
        
    def test_dataset_getitem(self):
        inputs, outputs = problem_dataset[0]
        self.assertEqual(len(inputs), len(df.columns))
        self.assertEqual(len(outputs), 2)

    def test_dataset_n_features(self):
        self.assertEqual(problem_dataset.n_features, len(df.columns))

    def test_dataset_row_column(self):
        self.assertFalse('Row' in problem_dataset.dataset.columns)
        self.assertFalse('Row' in problem_dataset.y.columns)
    
    def test_dataset_z_columns(self):
        self.assertFalse('z_1' in problem_dataset.dataset.columns)
        self.assertFalse('z_2' in problem_dataset.dataset.columns)
        self.assertTrue('z_1' in problem_dataset.y.columns)
        self.assertTrue('z_2' in problem_dataset.y.columns)


if __name__ == '__main__':
    unittest.main()