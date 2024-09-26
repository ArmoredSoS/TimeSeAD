import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from .dataset import BaseTSDataset
from typing import Tuple, Union, Any, Dict, List
from timesead.utils.metadata import DATA_DIRECTORY

import torch
import h5py
import os
import numpy

class CsesDataset(BaseTSDataset):

    def __init__(self, path: str = os.path.join(DATA_DIRECTORY, 'CSES'), training: bool = True):
        
        if not os.path.exists(path):
            raise NotADirectoryError
        self.path = path

        if not os.path.exists(os.path.join(self.path, 'training')):
            raise NotADirectoryError
        self.train_dir = os.path.join(self.path, 'training')

        if not os.path.exists(os.path.join(self.path, 'test')):
            raise NotADirectoryError
        self.test_dir = os.path.join(self.path, 'test')

        self.training = training
        self.ts_length = []
        self.ds_train = []
        self.ds_test = []
        self.load_data()

    def load_data(self):

        if self.training:

            for file in os.listdir(self.train_dir):

                E_X = []
                E_Y = []
                E_Z = []

                try:
                    with h5py.File(os.path.join(self.train_dir, file), 'r') as current_file:
                        E_X = current_file['A111_W'][:]
                        E_Y = current_file['A112_W'][:]
                        E_Z = current_file['A113_W'][:]
                except Exception as e:
                    print(e)
                    continue

                self.ts_length.append(min(len(E_X),len(E_Y),len(E_Z)))
                E_X = numpy.array(E_X, dtype=int)
                E_Y = numpy.array(E_Y, dtype=int)
                E_Z = numpy.array(E_Z, dtype=int)
                self.ds_train.append((E_X, E_Y, E_Z))

            return self.ds_train
        
        for file in os.listdir(self.test_dir):

            E_X = []
            E_Y = []
            E_Z = []

            try:
                with h5py.File(os.path.join(self.test_dir, file), 'r') as current_file:
                    E_X = current_file['A111_W'][:]
                    E_Y = current_file['A112_W'][:]
                    E_Z = current_file['A113_W'][:]
            except Exception as e:
                    print(e)
                    continue

            self.ts_length.append(min(len(E_X),len(E_Y),len(E_Z)))
            E_X = (numpy.array(E_X, dtype=int))
            E_Y = (numpy.array(E_Y, dtype=int))
            E_Z = (numpy.array(E_Z, dtype=int))
            self.ds_test.append((E_X, E_Y, E_Z))

        return self.ds_test
    
    def __len__(self) -> int:

        if self.training:
            return len(self.ds_train)
        return len(self.ds_test)
            
    @property
    def seq_len(self) -> Union[int, List[int]]:
        return self.ts_length
    
    @property   
    def num_features(self) -> Union[int, Tuple[int, ...]]:
        return 3
    
    @staticmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        return {
            'identity': {
                'class': 'NoTransform',
                'args': {}
            }
        }
    
    @staticmethod
    def get_feature_names() -> List[str]:
        return ['E_X', 'E_Y', 'E_Z']
    
    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:

        if not (0 <= index < len(self)):
            raise KeyError('Out of bounds')

        if self.ds_test is None or self.ds_train is None:
            self.load_data()

        if not self.training:
            return torch.as_tensor(self.ds_test[index]), index

        return torch.as_tensor(self.ds_train[index]), index