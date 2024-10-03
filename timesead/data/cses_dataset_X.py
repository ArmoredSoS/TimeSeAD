import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from .dataset import BaseTSDataset
from typing import Tuple, Union, Any, Dict, List
from timesead.utils.metadata import DATA_DIRECTORY
from sklearn.preprocessing import StandardScaler

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
        self.files_test = []
        self.files_train = []
        self.scaler = StandardScaler()

        if not training:
            self.files_test = os.listdir(self.test_dir)
        else:
            self.files_train = os.listdir(self.train_dir)
        
        self.ts_length = []

    def load_data(self, filepath):
        
        E_X = numpy.zeros((1100, 256))
        E_X_res = numpy.zeros((1, 1100, 1))

        try:
            with h5py.File(filepath, 'r') as current_file:

                E_X = numpy.array(current_file['A111_W'][:], dtype=int)
                #E_X = E_X[300:]      

                E_X_res = numpy.mean(E_X, axis=1)
                E_X_res = numpy.pad(E_X_res, (0, 1100 - len(E_X)), mode='constant', constant_values=0)    
                E_X_res = numpy.expand_dims(E_X_res, axis=(0,2))
                        
        except Exception as e:
            print(e)
        
        if E_X_res.all:
            return E_X_res
    
    def __len__(self) -> int:

        if self.training:
            return len(self.files_train)
        return len(self.files_test)
            
    @property
    def seq_len(self) -> Union[int, List[int]]:
        return 1100
    
    @property   
    def num_features(self) -> Union[int, Tuple[int, ...]]:
        return 1
    
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
        return ['E_X']
    
    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:

        if self.training:
            dir = self.train_dir
            file = self.files_train[index]
        else:
            dir = self.test_dir
            file = self.files_test[index]

        filepath = os.path.join(dir, file)
        
        return torch.as_tensor(self.load_data(filepath),), index