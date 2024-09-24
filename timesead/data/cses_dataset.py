import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from dataset import BaseTSDataset
from typing import Tuple, Union, Any, Dict, List
from timesead.utils.metadata import DATA_DIRECTORY

import torch
import h5py
import pandas
import os

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
        self.ts_length = None
        self.ds_train = None
        self.ds_test = None

    def load_data(self):

        E_X = []
        E_Y = []
        E_Z = []
        time = []

        if self.training:
            for file in os.listdir(self.train_dir):
                with h5py.File(file, 'r') as current_file:
                    E_X.append(current_file['A111_W'][:])
                    E_Y.append(current_file['A112_W'][:])
                    E_Z.append(current_file['A113_W'][:])
                    time.append(current_file['VERSE_TIME'][:])
                self.ts_length.append(len(E_X))
                self.ds_train = pandas.DataFrame({'time': time, 'E_X':E_X, 'E_Y':E_Y, 'E_Z':E_Z}).set_index('time')
            return self.ds_train
        
        for file in os.listdir(self.test_dir):
            with h5py.File(file, 'r') as current_file:
                E_X.append(current_file['A111_W'][:])
                E_Y.append(current_file['A112_W'][:])
                E_Z.append(current_file['A113_W'][:])
                time.append(current_file['VERSE_TIME'][:])
            self.ts_length.append(len(E_X))
            self.ds_test = pandas.DataFrame({'time': time, 'E_X':E_X, 'E_Y':E_Y, 'E_Z':E_Z}).set_index('time')
        return self.ds_test

    def __len__(self) -> int:

        if self.training:
            return len(self.train_dir)
        return len(self.test_dir)
            
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
            return (torch.as_tensor(self.ds_test[index]))

        return (torch.as_tensor(self.ds_train[index]))
    
    
    
    
