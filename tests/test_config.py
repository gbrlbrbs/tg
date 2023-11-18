import unittest
import schema
from pathlib import Path
from pyhardgen.config import load_exp, Config

class TestConfig(unittest.TestCase):

    def test_load_config(self):
        path = Path('files/test_config.yaml')
        config = load_exp(path)
        self.assertIsInstance(config, Config)
        self.assertEqual(config.name, 'test')
        self.assertEqual(config.ngpu, 1)
        self.assertEqual(config.epochs, 10)
        self.assertEqual(config.runs, 1)
        self.assertEqual(config.workers, 1)
        self.assertEqual(config.nn.n_layers, 3)
        self.assertEqual(config.nn.features, [4, 3, 2])
        self.assertEqual(config.nn.activation, 'relu')
        self.assertEqual(config.nn.loss, 'mse')
        self.assertEqual(config.dataset.filename, 'data.csv')
        self.assertEqual(config.dataset.batch_size, 32)
        self.assertEqual(config.optimizer.name, 'adam')
        self.assertEqual(config.optimizer.lr, 0.001)
        self.assertEqual(config.optimizer.weight_decay, 0.0000001)

    def test_config_schema_error(self):
        path = Path('files/test_config_schema_error.yaml')
        with self.assertRaises(schema.SchemaError):
            load_exp(path)

if __name__ == '__main__':
    unittest.main()